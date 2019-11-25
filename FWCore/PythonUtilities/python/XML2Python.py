from __future__ import print_function
## Original version of code heavily based on recipe written by Wai Yip
## Tung, released under PSF license.
## http://code.activestate.com/recipes/534109/

import re
import os
import xml.sax.handler
import pprint
import six

class DataNode (object):

    spaces = 4

    def __init__ (self, **kwargs):
        self._attrs = {}     # XML attributes and child elements
        self._data  = None   # child text data
        self._ncDict = kwargs.get ('nameChangeDict', {})


    def __len__ (self):
        # treat single element as a list of 1
        return 1


    def __getitem__ (self, key):
        if isinstance (key, str):
            return self._attrs.get(key,None)
        else:
            return [self][key]


    def __contains__ (self, name):
        return name in self._attrs


    def __nonzero__ (self):
        return bool (self._attrs or self._data)


    def __getattr__ (self, name):
        if name.startswith('__'):
            # need to do this for Python special methods???
            raise AttributeError (name)
        return self._attrs.get (name, None)


    def _add_xml_attr (self, name, value):
        change = self._ncDict.get (name)
        if change:
            name = change
        if name in self._attrs:
            # multiple attribute of the same name are represented by a list
            children = self._attrs[name]
            if not isinstance(children, list):
                children = [children]
                self._attrs[name] = children
            children.append(value)
        else:
            self._attrs[name] = value


    def __str__ (self):
        return self.stringify()


    def __repr__ (self):
        items = sorted (self._attrs.items())
        if self._data:
            items.append(('data', self._data))
        return u'{%s}' % ', '.join([u'%s:%s' % (k,repr(v)) for k,v in items])


    def attributes (self):
        return self._attrs


    @staticmethod
    def isiterable (obj):
        return getattr (obj, '__iter__', False)


    @staticmethod
    def _outputValues (obj, name, offset):
        retval = ' ' * offset
        if name:
            retval += '%s: ' % name
            offset += len (name) + DataNode.spaces
        # if this is a list
        if isinstance (obj, list):
            first = True
            for value in obj:
                print("value", value, value.__class__.__name__)
                if first:
                    tempoffset = offset
                    first = False
                    retval += '[\n ' + ' ' * offset
                else:
                    retval += ',\n ' + ' ' * offset
                    tempoffset = offset
                if isinstance (value, DataNode):
                    retval += value.stringify (offset=tempoffset)
                    print("  calling stringify for %s" % value)
                elif DataNode.isiterable (value):
                    retval += DataNode._outputValues (value, '', offset)
                else:
                    retval += "%s" % value
            retval += '\n' + ' ' * (offset - 2) +']\n'
            return retval
        retval += pprint.pformat(obj,
                                 indent= offset,
                                 width=1)
        return retval


    def stringify (self, name = '', offset = 0):
        # is this just data and nothing below
        if self._data and not len (self._attrs):
            return _outputValues (self._data, name, offset)
            retval = ' ' * offset
            if name:
                retval += '%s : %s\n' % \
                          (name,
                           pprint.pformat (self._data,
                                          indent= offset+DataNode.spaces,
                                          width=1) )
            else:
                retval += pprint.pformat (self._data,
                                          indent=offset+DataNode.spaces,
                                          width=1)
            return retval
        # this has attributes
        retval = ''
        if name:
            retval += '\n' + ' ' * offset
            retval += '%s: ' % name
        first = True
        for key, value in sorted (six.iteritems(self._attrs)):
            if first:
                retval += '{ \n'
                tempspace = offset + 3
                first = False
            else:
                retval += ',\n'
                tempspace = offset + 3
            if isinstance (value, DataNode):
                retval += value.stringify (key, tempspace)
            else:
                retval += DataNode._outputValues (value, key, tempspace)
        # this has data too
        if self._data:
            retval += ',\n'
            tempspace = offset + 3
            retval += DataNode._ouptputValues (self._data, name, tempspace)
        retval += '\n ' + ' ' * offset + '}'
        return retval 
        


class TreeBuilder (xml.sax.handler.ContentHandler):

    non_id_char = re.compile('[^_0-9a-zA-Z]')

    def __init__ (self, **kwargs):
        self._stack = []
        self._text_parts = []
        self._ncDict = kwargs.get ('nameChangeDict', {})
        self._root = DataNode (nameChangeDict = self._ncDict)
        self.current = self._root

    def startElement (self, name, attrs):
        self._stack.append( (self.current, self._text_parts))
        self.current = DataNode (nameChangeDict = self._ncDict)
        self._text_parts = []
        # xml attributes --> python attributes
        for k, v in attrs.items():
            self.current._add_xml_attr (TreeBuilder._name_mangle(k), v)

    def endElement (self, name):
        text = ''.join (self._text_parts).strip()
        if text:
            self.current._data = text
        if self.current.attributes():
            obj = self.current
        else:
            # a text only node is simply represented by the string
            obj = text or ''
        self.current, self._text_parts = self._stack.pop()
        self.current._add_xml_attr (TreeBuilder._name_mangle(name), obj)

    def characters (self, content):
        self._text_parts.append(content)

    def root (self):
        return self._root

    def topLevel (self):
        '''Returns top level object'''
        return list(self._root.attributes().values())[0]
        

    @staticmethod
    def _name_mangle (name):
        return TreeBuilder.non_id_char.sub('_', name)


regexList = [ (re.compile (r'&'), '&amp;'   ),
              (re.compile (r'<'), '&lt;'    ),
              (re.compile (r'>'), '&gt;'    ),
              (re.compile (r'"'), '&quote;' ),
              (re.compile (r"'"), '&#39;'   )
              ]

quoteRE = re.compile (r'(\w\s*=\s*")([^"]+)"')

def fixQuoteValue (match):
    '''Changes all characters inside of the match'''
    quote = match.group(2)
    for regexTup in regexList:
        quote = regexTup[0].sub( regexTup[1], quote )
    return match.group(1) + quote + '"'


def xml2obj (**kwargs):
    ''' Converts XML data into native Python object.  Takes either
    file handle or string as input.  Does NOT fix illegal characters.

    input source:  Exactly one of the three following is needed
    filehandle     - input from file handle
    contents       - input from string
    filename       - input from filename

    options:
    filtering      - boolean value telling code whether or not to fileter
                     input selection to remove illegal XML characters
    nameChangeDict - dictionaries of names to change in python object'''

    # make sure we have exactly 1 input source
    filehandle = kwargs.get ('filehandle')
    contents   = kwargs.get ('contents')
    filename   = kwargs.get ('filename')
    if not filehandle and not contents and not filename:
        raise RuntimeError("You must provide 'filehandle', 'contents', or 'filename'")
    if     filehandle and contents or \
           filehandle and filename or \
           contents   and filename:
        raise RuntimeError("You must provide only ONE of 'filehandle', 'contents', or 'filename'")

    # are we filtering?
    filtering = kwargs.get ('filtering')
    if filtering:
        # if we are filtering, we need to read in the contents to modify them
        if not contents:
            if not filehandle:
                try:
                    filehandle = open (filename, 'r')
                except:
                    raise RuntimeError("Failed to open '%s'" % filename)
            contents = ''
            for line in filehandle:
                contents += line
            filehandle.close()
            filehandle = filename = ''
        contents = quoteRE.sub (fixQuoteValue, contents)
    
    ncDict = kwargs.get ('nameChangeDict', {})
    builder = TreeBuilder (nameChangeDict = ncDict)
    if contents:
        xml.sax.parseString(contents, builder)
    else:
        if not filehandle:
            try:
                filehandle = open (filename, 'r')
            except:
                raise RuntimeError("Failed to open '%s'" % filename)
        xml.sax.parse(filehandle, builder)
    return builder.topLevel()
