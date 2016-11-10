
from ROOT import TEllipse, TBox
from ROOT import TColor, kRed, kBlue, kCyan

#TODO display the field
#TODO display trajectories (tracks, particles, charged or not)
#TODO display deposits


COLORS = dict(
    ECAL = kRed-10,
    HCAL = kBlue-10,
    void = None,
    BeamPipe = kCyan-10
) 

class GDetectorElement(object):
    '''TODO improve design? 
    there could be one detector element per view, 
    and they would all be linked together. 
    '''
    def __init__(self, description):
        self.desc = description
        self.circles = []
        self.boxes = []
        self.circles.append( TEllipse(0., 0.,
                                      self.desc.volume.outer.rad,
                                      self.desc.volume.outer.rad) )
        dz = self.desc.volume.outer.z
        radius = self.desc.volume.outer.rad
        self.boxes.append( TBox(-dz, -radius, dz, radius) ) 
        
        if self.desc.volume.inner:
            self.circles.append( TEllipse(0., 0.,
                                          self.desc.volume.inner.rad,
                                          self.desc.volume.inner.rad))
            dz = self.desc.volume.inner.z
            radius = self.desc.volume.inner.rad
            self.boxes.append( TBox(-dz, -radius, dz, radius) ) 
        color = COLORS[self.desc.material.name]
        oc = self.circles[0]
        ob = self.boxes[0]
        for shape in [oc, ob]:
            if color: 
                shape.SetFillColor(color)
                shape.SetFillStyle(1001)
            else:
                shape.SetFillStyle(0)
            shape.SetLineColor(1)
            shape.SetLineStyle(1)                
        if len(self.circles)==2:
            ic = self.circles[1]
            ib = self.boxes[1]
            for shape in [ic, ib]:
                if color:
                    shape.SetFillColor(0)
                    shape.SetFillStyle(1001)
                else:
                    shape.SetFillStyle(0)
                
    def draw(self, projection):
        if projection == 'xy':
            for circle in self.circles:
                circle.Draw('same')
        elif projection in ['xz', 'yz']:
            for box in self.boxes:
                box.Draw('samel')
        elif 'thetaphi' in projection:
            pass
        else:
            raise ValueError('implement drawing for projection ' + projection )


class GDetector(object):
    def __init__(self, description):
        self.desc = description
        elems = sorted(self.desc.elements.values(), key= lambda x : x.volume.outer.rad, reverse = True)
        self.elements = [GDetectorElement(elem) for elem in elems]
        #self.elements = [GDetectorElement(elem) for elem in self.desc.elements.values()]
            
    def draw(self, projection):
        for elem in self.elements:
            elem.draw(projection)


            
if __name__ == '__main__':

    from ROOT import TCanvas, TH2F
    from PhysicsTools.HeppyCore.papas.detectors.CMS import CMS
    from PhysicsTools.HeppyCore.display.core import Display

    cms = CMS()
    gcms = GDetector(cms)

    display = Display()
    display.register(gcms, 0)
    display.draw()
