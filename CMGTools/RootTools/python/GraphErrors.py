from ROOT import TGraphErrors,TH2F

import re

class GraphErrors( TGraphErrors ):
    def __init__(self, name, file, pattern):
        self.name = name
        self.parseFile(file, pattern)
        self.initGraph()
##         self.histo = None
        
    def initGraph(self):
        TGraphErrors.__init__(self, len(self.x) )
        i = 0
        for i in range(0, len(self.x) ):
            # print i, self.x[i], self.y[i]
            self.SetPoint(i, self.x[i], self.y[i])
            errx = 0
            erry = 0
            if len(self.errx)>i:
                errx = self.errx[i]
            if len(self.erry)>i:
                erry = self.erry[i]
            self.SetPointError(i, errx, erry)

##     def initHistogram(self):
##         minX = min(self.x)
##         minY = min(self.y)
##         maxX = max(self.x)
##         maxY = max(self.y)

##         maxErrY = 0
##         if len(self.erry):
##             maxErrY = max(self.erry)
        
##         xMargin = (maxX - minX)/10.
##         yMargin = (maxY - minY)/10. + maxErrY
        
##         self.histo = TH2F(self.name, '',
##                           100, minX-xMargin, maxX + xMargin,
##                           100, minY-yMargin, maxY + yMargin)
##         self.histo.SetStats(0)
        
##     def draw(self, opt=None):
##         if opt!='same':
##             if self.histo == None:
##                 self.initHistogram()
##             self.histo.Draw()
##         self.Draw('same')
    
    def parseFile(self, file, pattern):
        self.x = []
        self.errx = []
        self.y = []
        self.erry = []
        
        self.file = file
        input = open( file, 'r' )
        pattern = pattern.replace('FLOAT','-*\d+[.\d*]*')
        # print pattern
        pat = re.compile(pattern)
        if pat.groups<2 or pat.groups>4:
            print 'GraphErrors.parseFile : need between 2 and 4 groups:'
            print 'x y'
            print 'x y erry'
            print 'x errx y erry'
            return False
        for line in input.readlines():
            # print line
            match = pat.match( line )
            if match!=None:
                # print 'match!'
                if pat.groups >1:
                    # print 'x1', match.group(1)
                    self.x.append( float(match.group(1)) )
                if pat.groups <4:
                    # print 'y2', match.group(2)
                    self.y.append( float(match.group(2)) )
                if pat.groups == 3:
                    # print 'erry3', match.group(3)
                    self.erry.append( float(match.group(3)) )
                if pat.groups == 4:
                    self.errx.append( float(match.group(2)) )
                    self.y.append( float(match.group(3)) )
                    self.erry.append( float(match.group(4)) )
                    # print 'errx2', match.group(2)
                    # print 'y3', match.group(3)
                    # print 'erry4', match.group(4)
        return True
                

if __name__ == '__main__':
    import sys
    from CMGTools.RootTools.Style import *
    graph = GraphErrors('toto', sys.argv[1],'\s*(FLOAT)\s+(FLOAT)\s+(FLOAT)\s*')
    sBlueSquares.formatHisto( graph )
    graph.Draw('APL')
