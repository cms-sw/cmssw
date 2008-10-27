# A simple HTML table parser. It turns tables (including nested tables) into arrays
# Nigel Sim <nigel.sim@gmail.com>
# http://simbot.wordpress.com
from HTMLParser import HTMLParser
import re, string, os
from string import lower

class Table(list):
    pass
	
class Row(list):
    pass

class Cell(object):
    def __init__(self):
        self.data = None
        return
    def append(self,item):
        if self.data != None:
	    print "Overwriting %s"%self.data
        self.data = item

# Get the item on the top of a stack
def top(x):
    return x[len(x)-1]

class TableParser(HTMLParser):
    def __init__(self, parser=None):
        """
	The parser is a method which will be passed the doc at the end
	of the parsing. Useful if TableParser is within an inner loop and
	you want to automatically process the document. If it is omitted then
	it will do nothing
	"""
        self._tag = None
	self._buf = None
	self._attrs = None
	self.doc = None # Where the document will be stored
	self._stack = None
	self._parser = parser
	self.reset()
        return

    def reset(self):
        HTMLParser.reset(self)
	self.doc = []
	self._stack = [self.doc]
	self._buf = ''

    def close(self):
        HTMLParser.close(self)
	if self._parser != None:
	    self._parser(self.doc)

    def handle_starttag(self, tag, attrs):
        self._tag = tag
	self._attrs = attrs
	if lower(tag) == 'table':
	    self._buf = ''
            self._stack.append(Table())
	elif lower(tag) == 'tr':
	    self._buf = ''
            self._stack.append(Row())
	elif lower(tag) == 'td':
	    self._buf = ''
            self._stack.append(Cell())
	
        #print "Encountered the beginning of a %s tag" % tag

    def handle_endtag(self, tag):
	if lower(tag) == 'table':
	    t = None
	    while not isinstance(t, Table):
                t = self._stack.pop()
	    r = top(self._stack)
            r.append(t)

	elif lower(tag) == 'tr':
	    t = None
	    while not isinstance(t, Row):
                t = self._stack.pop()
	    r = top(self._stack)
            r.append(t)

	elif lower(tag) == 'td':
	    c = None
	    while not isinstance(c, Cell):
                c = self._stack.pop()
	    t = top(self._stack)
	    if isinstance(t, Row):
	        # We can not currently have text and other table elements in the same cell. 
		# Table elements get precedence
	        if c.data == None:
                    t.append(self._buf)
		else:
		    t.append(c.data)
	    else:
	        print "Cell not in a row, rather in a %s"%t
        self._tag = None
        #print "Encountered the end of a %s tag" % tag

    def handle_data(self, data):
        self._buf += data
