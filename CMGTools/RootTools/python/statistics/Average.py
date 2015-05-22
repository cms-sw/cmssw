#ROOTTOOLS
import math
import pickle
from CMGTools.RootTools.utils.diclist import diclist

class Average(object):
    def __init__(self, name):
        self.name = name
        self.sumw = 0
        self.sumwx = 0
        self.sumwx2 = 0
        # self.values = []
        
    def add(self, value, weight=1.0):
        self.sumw += weight
        self.sumwx += weight * value
        self.sumwx2 += weight * value * value
        # self.values.append( (value, weight) )        

    def variance(self):
        return abs( self.sumwx2 / self.sumw - \
                    self.sumwx * self.sumwx / (self.sumw*self.sumw) ) 
    
    def average( self ):
        ave = None
        err = None 
        if self.sumw:
            ave = self.sumwx / self.sumw
            # print self.sumwx, self.sumw, self.variance()
            err = math.sqrt( self.variance() ) / math.sqrt( self.sumw ) 
        return ave, err

    def __add__(self, other):
        '''Add two averages (+).'''
        self.sumw += other.sumw
        self.sumwx += other.sumwx
        self.sumwx2 += other.sumwx2
        return self

    def __iadd__(self, other):
        '''Add two averages (+=).'''
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
        ave, err = self.average()
        tmp = None
        if ave is not None:
            tmp = 'Average {name:<15}: {average: 8.4f} +- {err:8.4f}'
            # tmp = 'Average {name:<15}: {average: 8.4f}'
            tmp = tmp.format( name = self.name,
                              average = ave,
                              err = err
                              )
        else:
            tmp = 'Average {name:<15}: undefined (call Average.add)'.format( name = self.name)
        return tmp


class Averages(object):
    
    def __init__( self ):
        self.averages = []
        self.ranks = {}
        
    def addAverage(self, name):
        self.ranks[ name ] = len( self.averages )
        self.averages.append( Average(name) ) 

    def average(self, name):
        return self.averages[ self.ranks[name] ] 


class Averages(diclist):
    def write(self, dirname):
        map( lambda x: x.write(dirname), self)


if __name__ == '__main__':
    c = Average('TestAve')
    c.add( 1, 1 )
    c.add( 3, 2 )
    print c.variance()

    c2 = Average('TestAve2')
    # c2.add(10,1)

    sum = c+c2
    print c
    print c2
    print sum
    sum.write('.')

    import random

    c3 = Average('Gauss')
    for i in range(0,1000):
        c3.add( random.gauss( 5, 1 ) ) 
    print c3
    # print math.sqrt( c3.variance(c3.average()[0]) )
