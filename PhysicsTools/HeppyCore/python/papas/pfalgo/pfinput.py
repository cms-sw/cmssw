import operator

def merge_clusters(clusters):
    pass

class PFInput(object):
    '''Builds the inputs to particle flow from a collection of simulated particles:
    - collects all smeared tracks and clusters
    - merges overlapping clusters 
    '''
    
    def __init__(self, ptcs):
        '''
        attributes: 
        - elements: dictionary of elements:
          tracker : [track0, track1, ...]
          ecal: [cluster0, cluster1, ...]
          hcal: [... ]
        '''
        self.elements = dict()
        self.build(ptcs)
        
    def build(self, ptcs):
        for ptc in ptcs:
            for key, cluster in ptc.clusters_smeared.iteritems():
                self.elements.setdefault(key, []).append(cluster)
            if ptc.track_smeared:
                self.elements.setdefault('tracker', []).append(ptc.track_smeared)

        #Alice disabled sort
        #for elems in self.elements.values():
        #    elems.sort(key=operator.attrgetter('energy'), reverse=True)

    def element_list(self):
        thelist = []
        for layer, elements in sorted(self.elements.iteritems()):
            thelist.extend( elements )
        return thelist
            
    def __str__(self):
        lines = ['PFInput:']
        # lines.append('\tTracks:')
        def tab(astr, ntabs=2):
            return ''.join(['\t'*ntabs, str(astr)])
        # for track in self.tracks:
        #     lines.append(tab(str(track)))
        # lines.append('\tClusters:')
        for layer, elements in sorted(self.elements.iteritems()):
            lines.append(tab(layer))
            for element in elements:
                lines.append(tab(str(element), 3))
        return '\n'.join(lines)
