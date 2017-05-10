import unittest
import os
import logging
import pdebug as pdebug
from StringIO import StringIO

class TestPDebug(unittest.TestCase):

    def test_debug_output(self):
        out = StringIO()
        pdebug.pdebugger.setLevel(logging.ERROR)
        pdebug.set_stream(out)
        pdebug.pdebugger.error('error console')
        pdebug.pdebugger.info('info console')
        pdebug.pdebugger.info('debug console')
        output = out.getvalue().strip()
        assert output == "error console"


        pdebug.pdebugger.setLevel(logging.INFO)
        pdebug.pdebugger.error('error console')
        pdebug.pdebugger.info('info console')
        pdebug.pdebugger.debug('debug console')
        output = out.getvalue().strip()
        assert output == "error console\nerror console\ninfo console"


        #add in file handler
        filename = "tempunittestdebug.log"
        pdebug.set_file(filename)
        pdebug.pdebugger.error('error file')
        pdebug.pdebugger.info('info file')
        pdebug.pdebugger.debug('debug file')
        with open(filename, 'r') as dbfile:
            data=dbfile.read()
        assert data == "error file\ninfo file\n"
        os.remove("tempunittestdebug.log")
        output = out.getvalue().strip()
        assert output == "error console\nerror console\ninfo console\nerror file\ninfo file"


if __name__ == '__main__':

    unittest.main()
