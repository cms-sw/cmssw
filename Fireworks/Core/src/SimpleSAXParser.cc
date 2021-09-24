#include "Fireworks/Core/interface/SimpleSAXParser.h"

/** Helper function to handle entities, i.e. characters specified with
    the "&label;" syntax.
  */
std::string SimpleSAXParser::parseEntity(const std::string &entity) {
  if (entity == "quot")
    return "\"";
  else if (entity == "amp")
    return "&";
  else if (entity == "lt")
    return "<";
  else if (entity == "gt")
    return ">";
  throw ParserError("Unknown entity " + entity);
}

void debug_state_machine(enum SimpleSAXParser::PARSER_STATES state) {
#ifdef SIMPLE_SAX_PARSER_DEBUG
  static char *debug_states[] = {"IN_DOCUMENT",
                                 "IN_BEGIN_TAG",
                                 "IN_DONE",
                                 "IN_BEGIN_ELEMENT",
                                 "IN_ELEMENT_WHITESPACE",
                                 "IN_END_ELEMENT",
                                 "IN_ATTRIBUTE_KEY",
                                 "IN_END_TAG",
                                 "IN_DATA",
                                 "IN_BEGIN_ATTRIBUTE_VALUE",
                                 "IN_STRING",
                                 "IN_END_ATTRIBUTE_VALUE",
                                 "IN_STRING_ENTITY",
                                 "IN_DATA_ENTITY"};

  std::cerr << debug_states[state] << std::endl;
#endif
}

/** Runs the state machine of the parser, invoking startElement(),
    setAttribute(), endElement(), data() virtual methods as approppriate. 
    In order have the parser doing something usefull you need to derive from
    it and specialize the above mentioned virtual methods.
    
    Default implementation is in any case useful to check syntax.
  */
void SimpleSAXParser::parse(void) {
  enum PARSER_STATES state = IN_DOCUMENT;
  // Current delimiters for strings in attributes.
  char stringDelims[] = "\"&";
  std::string attributeName;
  std::string attributeValue;
  std::string tmp;
  std::string currentData;

  while (state != IN_DONE) {
    debug_state_machine(state);

    switch (state) {
      // FIXME: IN_DOCUMENT should check the dtd...
      case IN_DOCUMENT:
        state = IN_DATA;
        if (skipChar('<'))
          state = IN_BEGIN_TAG;
        break;

      case IN_BEGIN_TAG:
        if (nextChar() >= 'A' && nextChar() <= 'z')
          state = IN_BEGIN_ELEMENT;
        else if (skipChar('/'))
          state = IN_END_ELEMENT;
        else
          throw ParserError("Bad tag");
        break;

      case IN_BEGIN_ELEMENT:
        m_attributes.clear();
        m_elementTags.push_back(getToken(" />"));
        if (nextChar() == ' ')
          state = IN_ELEMENT_WHITESPACE;
        else if (skipChar('/'))
          state = IN_END_ELEMENT;
        else if (skipChar('>')) {
          startElement(m_elementTags.back(), m_attributes);
          state = IN_END_TAG;
        } else
          throw ParserError("Bad element.");
        break;

      case IN_ELEMENT_WHITESPACE:
        while (skipChar(' ') || skipChar('\n') || skipChar('\t')) {
        }

        if (nextChar() >= 'A' && nextChar() <= 'z')
          state = IN_ATTRIBUTE_KEY;
        else if (nextChar() == '/')
          state = IN_END_ELEMENT;
        else
          throw ParserError("Syntax error in element" + m_elementTags.back());
        break;

      case IN_ATTRIBUTE_KEY:
        attributeName = getToken('=');
        state = IN_BEGIN_ATTRIBUTE_VALUE;
        break;

      case IN_BEGIN_ATTRIBUTE_VALUE:
        if (skipChar('"')) {
          state = IN_STRING;
          attributeValue.clear();
          stringDelims[0] = '\"';
        } else if (skipChar('\'')) {
          state = IN_STRING;
          attributeValue.clear();
          stringDelims[0] = '\'';
        } else
          throw ParserError("Expecting quotes.");
        break;

      case IN_STRING:
        attributeValue += getToken(stringDelims);
        if (skipChar(stringDelims[0])) {
          // Save the attributes in order, replacing those that are
          // specified more than once.
          Attribute attr(attributeName, attributeValue);
          Attributes::iterator i = std::lower_bound(m_attributes.begin(), m_attributes.end(), attr);
          if (i != m_attributes.end() && i->key == attr.key)
            throw ParserError("Attribute " + i->key + " defined more than once");
          m_attributes.insert(i, attr);
          state = IN_END_ATTRIBUTE_VALUE;
        } else if (skipChar(stringDelims[1]))
          state = IN_STRING_ENTITY;
        else
          throw ParserError("Unexpected end of input at " + attributeValue);
        break;

      case IN_END_ATTRIBUTE_VALUE:
        getToken(" />");
        if (nextChar() == ' ')
          state = IN_ELEMENT_WHITESPACE;
        else if (skipChar('/'))
          state = IN_END_ELEMENT;
        else if (skipChar('>')) {
          startElement(m_elementTags.back(), m_attributes);
          state = IN_END_TAG;
        }
        break;

      case IN_END_ELEMENT:
        tmp = getToken('>');
        if (!tmp.empty() && tmp != m_elementTags.back())
          throw ParserError("Non-matching closing element " + tmp + " for " + attributeValue);
        endElement(tmp);
        m_elementTags.pop_back();
        state = IN_END_TAG;
        break;

      case IN_END_TAG:
        if (nextChar() == EOF)
          return;
        else if (skipChar('<'))
          state = IN_BEGIN_TAG;
        else
          state = IN_DATA;
        break;

      case IN_DATA:
        currentData += getToken("<&");
        if (skipChar('&'))
          state = IN_DATA_ENTITY;
        else if (skipChar('<')) {
          data(currentData);
          currentData.clear();
          state = IN_BEGIN_TAG;
        } else if (nextChar() == EOF) {
          data(currentData);
          return;
        } else
          throw ParserError("Unexpected end of input in element " + m_elementTags.back() + currentData);
        break;

      case IN_DATA_ENTITY:
        currentData += parseEntity(getToken(';'));
        state = IN_DATA;
        break;

      case IN_STRING_ENTITY:
        attributeValue += parseEntity(getToken(';'));
        state = IN_STRING;
        break;

      case IN_DONE:
        return;
    }
  }
}

SimpleSAXParser::~SimpleSAXParser() { delete[] m_buffer; }

/** Helper function which gets a token delimited by @a separator from the 
    @a file and write it, 0 terminated in the buffer found in @a buffer.
    
    Notice that if the token is larger than @a maxSize, the buffer is
    reallocated and @a maxSize is updated to the new size.

    The trailing separator after a token is not put in the token and is left 
    in the buffer. If @a nextChar is not 0, the delimiter is put there.
    
    @a in the input stream to be parsed.
    
    @a buffer a pointer to the buffer where to put the tokens. The buffer will
     be redimensioned accordingly, if the token is larger of the buffer.
     
    @a maxSize, a pointer to the size of the buffer. Notice that in case the 
     buffer is reallocated to have more space, maxSize is updated with the new 
     size.
     
    @a firstChar a pointer with the first character in the buffer, notice
                 that the first charater in the stream must be obtained 
                 separately!!!
    
    @return whether or not we were able to get a (possibly empty) token from
            the file.
  */
bool fgettoken(std::istream &in, char **buffer, size_t *maxSize, const char *separators, int *firstChar) {
  // if the passed first character is EOF or a separator,
  // return an empty otherwise use it as first character
  // of the buffer.
  if (*firstChar == EOF || (int)separators[0] == *firstChar || strchr(separators + 1, *firstChar)) {
    (*buffer)[0] = 0;
    return true;
  } else
    (*buffer)[0] = (char)*firstChar;

  size_t i = 1;

  while (true) {
    if (i >= *maxSize) {
      *maxSize += 1024;
      *buffer = (char *)realloc(*buffer, *maxSize);
      if (!*buffer)
        return false;
    }

    int c = in.get();

    if (c == EOF) {
      (*buffer)[i] = 0;
      *firstChar = c;
      return false;
    }

    if (separators[0] == c || strchr(separators + 1, c)) {
      (*buffer)[i] = 0;
      *firstChar = c;
      return true;
    }

    (*buffer)[i++] = (char)c;
  }
}
