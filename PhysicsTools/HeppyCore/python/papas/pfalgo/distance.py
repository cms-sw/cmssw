
class Distance(object):
    '''Concrete distance calculator.
    ''' 
    def __call__(self, ele1, ele2):
        '''returns a tuple: 
          True/False depending on the validity of the link
          float      the link distance
        '''
        layer1, layer2 = ele1.layer, ele2.layer
        if layer2 < layer1:
            layer1, layer2 = layer2, layer1
            ele1, ele2 = ele2, ele1
        layers = layer1, layer2
        func = None
        if layers == ('ecal_in', 'tracker'):
            func = self.ecal_track
        elif layers == ('hcal_in', 'tracker'):
            func = self.hcal_track
        elif layers == ('ecal_in', 'hcal_in'):
            func = self.no_link #Alice needed to make align with cpp ecal_hcal
        elif layers == ('ecal_in', 'ecal_in'):
            func = self.ecal_ecal
        elif layers == ('hcal_in', 'hcal_in'):
            func = self.hcal_hcal
        elif layers == ('tracker', 'tracker'):
            func = self.no_link
        else:
            raise ValueError('no such link layer:', layers)
        return func(ele1, ele2)        

    def no_link(self, ele1, ele2):
        return None, False, None
    
    def ecal_ecal(self, ele1, ele2):
        #modified this to also deal with clusters that are merged clusters
        link_ok, dist = ele1.is_inside_clusters(ele2)
        return ('ecal_in', 'ecal_in'), link_ok,  dist

    def hcal_hcal(self, ele1, ele2):
        link_ok, dist = ele1.is_inside_clusters(ele2)
        return ('hcal_in', 'hcal_in'), link_ok, dist 
    
    def ecal_track(self, ecal, track):
        tp = track.path.points.get('ecal_in', None)
        if tp is None:
            # probably a looper
            return ('ecal_in', 'tracker'), False, None
        link_ok, dist = ecal.is_inside(tp)
        return ('ecal_in', 'tracker'), link_ok, dist
        
    def hcal_track(self, hcal, track):
        tp = track.path.points.get('hcal_in', None)
        if tp is None:
            # probably a looper
            return ('hcal_in', 'tracker'), False, None
        link_ok, dist = hcal.is_inside(tp)
        return ('hcal_in', 'tracker'), link_ok, dist

    def ecal_hcal(self, ele1, ele2):
        link_ok, dist = ele1.is_inside_clusters(ele2)    
        return ('ecal_in', 'hcal_in'), link_ok, dist 

distance = Distance()
