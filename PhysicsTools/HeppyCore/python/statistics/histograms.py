# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

class Histograms(object):
    '''Base class to handle writing and formatting of a set of histograms. 

    Subclass it, and simply add your histograms to the subclass.
    No need to put them in a list, they will be kept track of automatically
    by this base class. 
    Then, fill them. Finally, you can call FormatHistos and Write.'''
    def __init__(self, name):
        self.name = name
        self.hists = []
        self.named = []
        # attributes inheriting from TH1 and TNamed
        # are kept track of automagically, even if they are in
        # child classes
        # setting StatOverflows True for all histograms
        for var in vars( self ).values():
            try:
                if var.InheritsFrom('TNamed'):
                    self.named.append(var)
                    if var.InheritsFrom('TH1'):
                        var.StatOverflows(True)
                        self.hists.append(var)
            except:
                pass
        # print 'TH1     list:', self.hists
        # print 'TNamed  list:', self.named

    def FormatHistos(self, style ):
        '''Apply a style to all histograms.'''
        for hist in self.hists:
            style.FormatHisto( hist )

    def Write(self, dir ):
        '''Writes all histograms to a subdirectory of dir called self.name.'''
        self.dir = dir.mkdir( self.name )
        self.dir.cd()
        for hist in self.hists:
            hist.Write()
        dir.cd()

        
