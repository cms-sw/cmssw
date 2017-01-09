# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import math
import copy

class Value(object):
    def __init__(self, val, err):
        self.val = val
        self.err = err

    def relerr(self):
        '''relative uncertainty. 

        returns None if value == 0 for __str__ to work.'''
        try:
            return abs(self.err / self.val)
        except ZeroDivisionError:
            return None

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
        relerr = self.relerr()
        relerr_format = '{relerr}'
        if relerr:
            relerr *= 100
            relerr_format = '{relerr:5.2f}%'
        format_template = '{{val:10.3f}} +- {{err:8.3f}} ({relerr_format})'.format(
            relerr_format = relerr_format
        )
        
        return format_template.format(val=self.val,
                                      err=self.err,
                                      relerr=relerr)
