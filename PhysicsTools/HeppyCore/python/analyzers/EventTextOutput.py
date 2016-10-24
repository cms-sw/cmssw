from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

import shelve

outfilename = 'particles.shv'

class SimpleParticle(object):
    def __init__(self, ptc):
        self.theta = ptc.theta()
        self.phi = ptc.phi()
        self.energy = ptc.e()
        self.pdgid = ptc.pdgid()
    
    def get_data(self):
        return self.pdgid, self.theta, self.phi, self.energy
        
    def __str__(self):
        return 'particle: {id} : theta={theta}, phi={phi}, energy={energy}'.format(
            id = self.pdgid,
            theta = self.theta,
            phi = self.phi,
            energy = self.energy
            )
    
class SimpleEvent(object):
    def __init__(self, ievent, ptcs):
        self.ievent = ievent
        self.ptcs = map(SimpleParticle, ptcs)
        self.data = dict(
            ievent = ievent,
            particles = [ptc.get_data() for ptc in self.ptcs]
            )

    def get_data(self):
        return self.data
        
    def __str__(self):
        lines = ['event {iev}'.format(iev=self.ievent)]
        lines.extend( map(str, self.ptcs) )
        return '\n'.join(lines)
        
        

class EventTextOutput(Analyzer):

    def beginLoop(self, setup):
        super(EventTextOutput, self).beginLoop(setup)
        self.events = []

    def process(self, event):
        ptcs = getattr(event, self.cfg_ana.particles )
        self.events.append(SimpleEvent(event.iEv, ptcs).get_data()) 
        
    def endLoop(self, setup):
        super(EventTextOutput, self).endLoop(setup)
        out = shelve.open('/'.join([self.dirName, outfilename]))
        out['events'] = self.events
        out.close()


if __name__ == '__main__':

    import pprint
    sh = shelve.open(outfilename)
    events = sh['events']
    for event in events:
        print 'event', event['ievent']
        for pdgid, theta, phi, energy  in event['particles']:
            print '\t', pdgid, theta, phi, energy
