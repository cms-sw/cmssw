from ROOT import TH1F
from CMGTools.RootTools.TaggedFile import *
import copy

class TagAndProbeOneLeg:
    def __init__(self, name, erropt = 'B'):
        self.name = name
        self.erropt = erropt
    
    def formatHistos(self, style, title=''):
        style.formatHisto( self.hTot, title)
        style.formatHisto( self.hSel, title)
        style.formatHisto( self.hEff, title)
        
    def fillHistos(self, events, var, nBins, min, max, probeCut, tagCut, nEvents):
        print 'var  : ', var
        print 'tag  : ', tagCut
        print 'probe: ', probeCut
        self.tagCut = tagCut
        self.probeCut = probeCut
        # the definition of the histogram should be external...
        self.hTot = TH1F('hTot_'+self.name,';'+var, nBins, min, max)
        self.hSel = self.hTot.Clone('hSel_'+self.name)
        print 'n_tot...'
        events.Draw(var + '>>hTot_'+self.name, tagCut,'goff', nEvents)
        print 'n_sel...'
        events.Draw(var + '>>hSel_'+self.name, tagCut + '&&' + probeCut,'goff', nEvents)
        print self.hSel.GetEntries(), '/', self.hTot.GetEntries()
        
    def efficiency(self): 
        self.hEff = self.hSel.Clone('hEff_'+self.name)
        self.hEff.SetTitle(';%s;Efficiency' % self.hSel.GetXaxis().GetTitle())
        self.hEff.SetStats(0)
        self.hEff.Sumw2()
        self.hTot.Sumw2()   
        self.hEff.Divide( self.hSel, self.hTot, 1, 1, self.erropt)
        
    def clone(self, other):
        self.hTot = other.hTot.Clone('hTot_'+self.name)
        self.hSel = other.hSel.Clone('hSel_'+self.name)
    def add(self, other):
        self.hTot.Add( other.hTot ) 
        self.hSel.Add( other.hSel)
        
    def write(self, dir):
        self.dir = dir.mkdir( self.name )
        self.dir.cd()
        self.hTot.Write()
        self.hSel.Write()
        self.hEff.Write()
        dir.cd()
        
    def load(self, dir):
        self.dir = dir.Get( self.name )
        self.hTot = self.dir.Get( 'hTot_' + self.name)
        self.hSel = self.dir.Get( 'hSel_' + self.name)
        self.hEff = self.dir.Get( 'hEff_' + self.name)

class TagAndProbeBothLegs:
    def __init__(self,name):
        self.name = name
        self.leg1 = TagAndProbeOneLeg( name + '_leg1')
        self.leg2 = TagAndProbeOneLeg( name + '_leg2')
        self.sum = TagAndProbeOneLeg( self.name + '_sum')
    
    def fillHistos(self, events, var, nBins, min, max,
                   probeCut, tagCut, nEvents ):
        
        print 'leg 1:'
        
        probeCut.leg = 'leg1'
        tagCut.leg = 'leg2'
        var = var.replace('leg2', 'leg1')
        self.leg1.fillHistos( events, var, nBins, min, max,
                              probeCut.__str__(), tagCut.__str__(), nEvents)    
        print 'leg 2:'
        probeCut.leg = 'leg2'
        tagCut.leg = 'leg1'
        var = var.replace('leg1', 'leg2')
        self.leg2.fillHistos( events, var, nBins, min, max,
                              probeCut.__str__(), tagCut.__str__(), nEvents)
        
        self.sum.clone(self.leg1)
        self.sum.add(self.leg2)
        self.efficiency()
        
        self.probeCut = probeCut
        self.tagCut = tagCut 
                
    def efficiency(self):
        self.leg1.efficiency()
        self.leg2.efficiency()
        self.sum.efficiency()
        
    def formatHistos(self, style, title=''):
        self.leg1.formatHistos(style, title)
        self.leg2.formatHistos(style, title)
        self.sum.formatHistos(style, title)
        
    def write(self):
        self.file = TaggedFile( self.name + '.root' )
        self.file.tag('probe', self.probeCut.__str__())
        self.file.tag('tag', self.tagCut.__str__())
        
        self.leg1.write( self.file.file )
        self.leg2.write( self.file.file )
        self.sum.write( self.file.file )
        
    def load(self, fileName ):
        self.file = TFile( fileName )
        self.leg1.load( self.file )
        self.leg2.load( self.file )
        self.sum.load( self.file )


        
