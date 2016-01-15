import sys

class Events(object):

    def __init__(self, dummy, dummy_2=None):
        pass

    def __len__(self):
        return sys.maxint

    def __getitem__(self, index):
        return self
