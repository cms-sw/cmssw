from CMGTools.RootTools.analyzers.TreeAnalyzer import TreeAnalyzer


class WNJetsTreeAnalyzer( TreeAnalyzer ):
    '''Requires WNJetsAnalyzer upstream.
    Fills a tree with the WJets event weight, and NUP (number of partons).
    '''
    def declareVariables(self):
        self.tree.addVar('float', 'wjetweight')
        self.tree.addVar('int', 'nup')
        self.tree.book()
        
    def process(self, event):
        self.tree.s.wjetweight = event.WJetWeight
        self.tree.s.nup = event.NUP
        self.tree.fill()
        
