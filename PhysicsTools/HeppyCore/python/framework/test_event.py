import unittest
import pprint
from event import Event

class LongPrintout(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'I have value {val} and a crazy, long printout that is really a pain to display'.format(val=self.value)


    
class EventTest(unittest.TestCase):

    def setUp(self):
        self.event = Event(0)
        self.event.a_list = range(10)
        self.event.a_list_2 = range(11)
        self.event.a_list_3 = range(12)
        self.event.list_of_long_stuff = map(LongPrintout, range(10))
        self.event.a_dict = dict([ (val, val**2) for val in range(10)])
        self.event.a_dict_2 = dict([ (val, LongPrintout(val)) for val in range(20)])
        self.event.setup = None
        self.event.input = 'data'
        
    def test_print(self):
        # Event.print_patterns = ['*list*']
        # print self.event
        str(self.event)
        self.assertTrue(True)
        
        
if __name__ == '__main__':
    unittest.main()
