# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import math
import copy

class Value(object):
    def __init__(self, val, err):
        self.val = val
        self.err = err

    def relerr(self):
        return abs(self.err / self.val)

    def __eq__(self, other):
        return self.val == other.val and self.err == other.err

    def __iadd__(self, other):
        self.val += other.val
        self.err = math.sqrt( self.err*self.err + other.err*other.err)
        return self

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new

    def __isub__(self, other):
        self.val -= other.val
        self.err = math.sqrt( self.err*self.err + other.err*other.err)
        return self

    def __sub__(self, other):
        new = copy.deepcopy(self)
        new -= other
        return new

    def __idiv__(self, other):
        relerr = math.sqrt( self.relerr()*self.relerr() + other.relerr()*other.relerr())
        self.val /= other.val
        self.err = relerr * self.val
        return self

    def __div__(self, other):
        new = copy.deepcopy(self)
        new /= other
        return new

    def __str__(self):
        return '{val:10.3f} +- {err:8.3f} ({relerr:5.2f}%)'.format(val=self.val,
                                                                  err=self.err,
                                                                  relerr=self.relerr()*100)
