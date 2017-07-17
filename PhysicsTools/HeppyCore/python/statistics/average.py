# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import math
import pickle
from PhysicsTools.HeppyCore.utils.diclist import diclist

class Average(object):

    def __init__(self, name):
        self.name = name
        self.sumw = 0
        self.sumwx = 0
        self.sumwx2 = 0

    def add(self, value, weight=1.0):
        """
        Add a new sample to the average.
        """
        value = float(value) # avoids surprising results with integers
        weight = float(weight)
        self.sumw += weight
        self.sumwx += weight * value
        self.sumwx2 += weight * value * value

    def variance(self):
        return abs( self.sumwx2 / self.sumw - \
                    self.sumwx * self.sumwx / (self.sumw*self.sumw) )

    def value(self):
        """
        Mean value
        """
        if self.sumw:
            return self.sumwx / self.sumw
        else:
            return None

    def uncertainty(self):
        """
        Uncertainty on the mean value
        """
        if self.sumw:
            return math.sqrt( self.variance() ) / math.sqrt( self.sumw )
        else:
            return None

    def average( self ):
        """
        Returns: mean value, uncertainty on mean value.
        """
        return self.value(), self.uncertainty()

    def __add__(self, other):
        '''Add two averages, merging the two samples.'''
        self.sumw += other.sumw
        self.sumwx += other.sumwx
        self.sumwx2 += other.sumwx2
        return self

    def __iadd__(self, other):
        '''Add two averages.'''
        return self.__add__(other)

    def write(self, dirname):
        '''Dump the average to a pickle file and to a text file in dirname.'''
        pckfname = '{d}/{f}.pck'.format(d=dirname, f=self.name)
        pckfile = open( pckfname, 'w' )
        pickle.dump(self, pckfile)
        txtfile = open( pckfname.replace('.pck', '.txt'), 'w')
        txtfile.write( str(self) )
        txtfile.write( '\n' )
        txtfile.close()

    def __str__(self):
        ave, unc = self.average()
        tmp = None
        if ave is not None:
            tmp = 'Average {name:<15}: {average: 8.4f} +- {unc:8.4f}'
            # tmp = 'Average {name:<15}: {average: 8.4f}'
            tmp = tmp.format( name = self.name,
                              average = ave,
                              unc = unc
                              )
        else:
            tmp = 'Average {name:<15}: undefined (call Average.add)'\
                  .format( name = self.name)
        return tmp



class Averages(diclist):
    def write(self, dirname):
        map( lambda x: x.write(dirname), self)
