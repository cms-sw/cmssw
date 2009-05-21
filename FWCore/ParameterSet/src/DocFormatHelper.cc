
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <ostream>
#include <iomanip>

namespace edm {

  void DocFormatHelper::init() {
    section_ = std::string();
    pass_ = 0;
    column1_ = 0;
    column2_ = 0;
    column3_ = 0;
    counter_ = 0;
    parent_ = OTHER;
  }

  size_t DocFormatHelper::commentWidth() const {

    // Make the length of a comment at least 30 characters
    // per line, longer if there is more space available
    size_t width = 30U;
    if (lineWidth() > startColumn2() + 30U) {
      width = lineWidth() - startColumn2();
    }
    return width;
  }

  // Print text to an output stream.  This function inserts
  // new lines to try to break the text to fit into the
  // suggested width.  At the beginning and after every
  // inserted newline this will print "indent" blank spaces.
  // The function will consider inserting a new line after
  // every blank space.  If the text to the next blank
  // space will exceed the "suggestedWidth" it inserts a
  // new line.  If the text between two blank spaces (a "word")
  // is longer than the suggested width, then it prints the whole
  // word anyway (exceeding the "suggestedWidth" for lines).
  // The output will look nice if the input has words separated
  // by a single blank space with no newlines or extra space in
  // the input text, otherwise the extra spaces and newlines get
  // printed making the output not nicely formatted ...
  void DocFormatHelper::wrapAndPrintText(std::ostream & os,
                                         std::string const& text,
                                         size_t indent,
                                         size_t suggestedWidth) {

    size_t length = text.size();

    // The position in the text where we start printing the next line
    size_t startLine = 0U;

    // The position in the text where we start looking for the next blank space
    size_t startNextSearch = 0U;

    // Loop over spaces in the text
    while (true) {

      // If the rest of the text fits on the current line,
      // then print it and we are done
      if ((length - startLine) <= suggestedWidth) {
        os << std::setfill(' ') << std::setw(indent) << "";
        if (startLine == 0) os << text; 
        else os << text.substr(startLine);
        os << "\n";
        break;
      }

      // Look for next space
      size_t pos = text.find_first_of(' ', startNextSearch);

      // No more spaces
      if (pos == std::string::npos) {
        // The rest of the comment cannot fit in the width or we
        // would have already printed it.  Break the line at the
        // end of the previous word if there is one and print the
        // first part.  Then print the rest whether it fits or not.
        if (startNextSearch != startLine) {
          os << std::setfill(' ') << std::setw(indent) << "";
          os << text.substr(startLine, startNextSearch - startLine);
          os << "\n";
          startLine = startNextSearch;
        }
        os << std::setfill(' ') << std::setw(indent) << "";
        os << text.substr(startLine);
        os << "\n";
        break;
      }

      if ((pos + 1U - startLine) > suggestedWidth) {

        // With this word the line is too long.  Print out to
        // the end of the previous word if there was one.
        // If there was not, then print to the end of this word
        // even though it exceeds the width.
        if (startNextSearch != startLine) {
          os << std::setfill(' ') << std::setw(indent) << "";
          os << text.substr(startLine, startNextSearch - startLine);
          os << "\n";
          startLine = startNextSearch;
        }
        if ((pos + 1U - startLine) > suggestedWidth) {
          os << std::setfill(' ') << std::setw(indent) << "";
          os << text.substr(startLine, pos + 1U - startLine);
          os << "\n";
          startLine = pos + 1U;
        }
      }
      startNextSearch = pos + 1U;
    }
  }

  void DocFormatHelper::indent(std::ostream & os) const {
    os << std::setfill(' ') << std::setw(indentation_) << "";
  }

  void DocFormatHelper::indent2(std::ostream & os) const {
    os << std::setfill(' ') << std::setw(startColumn2_) << "";
  }
}
