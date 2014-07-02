import ROOT
import array
import math
import copy

def deltaR2( e1, p1, e2, p2):
    de = e1 - e2
    dp = deltaPhi(p1, p2)
    return de*de + dp*dp


def deltaR( *args ):
    return math.sqrt( deltaR2(*args) )


def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = p1 - p2
    while res > math.pi:
        res -= 2*math.pi
    while res < -math.pi:
        res += 2*math.pi
    return res

class DisplayManager(object):
    def __init__(self,name,eta,phi,radius=0.5):
        self.etaPhiVew = ROOT.TGraph()
        self.etaPhiVew.SetName(name)
        self.rechits=[]
        self.genParticles=[]
        self.simParticles=[]
        self.tracks=[]
        self.clusters=[]
        self.clusterLinks=[]
        self.etaCenter=eta
        self.phiCenter=phi
        self.radius=radius
        self.name=name

    def markerByType(self,particle):
        if particle.pdgId()==22:
            return 4
        elif abs(particle.pdgId())==11:
            return 3
        elif abs(particle.pdgId())==13:
            return 5
        elif particle.charge()!=0:
            return 2
        elif particle.charge()==0:
            return 25
        else:
            return -1


    def markerBySimType(self,particle):
        if particle.pdgCode()==22:
            return 4
        else: 
            return 25
        

        
    def addRecHit(self,recHit,depth=1,fraction=1):
        if deltaR(recHit.position().Eta(),recHit.position().Phi(),self.etaCenter,self.phiCenter)>self.radius:
            return

        
        corners=[]
        for corner in recHit.getCornersXYZ():
            corners.append({'eta':corner.Eta(),'phi':corner.Phi(),'rho':corner.Rho()})


        rechit={'corners':corners,'energy':recHit.energy()*fraction,'depth':depth}
        self.rechits.append(rechit)

    def addCluster(self,cluster,links=False):
        if deltaR(cluster.position().Eta(),cluster.position().Phi(),self.etaCenter,self.phiCenter)>self.radius:
            return
        
        print cluster.energy(),cluster.position().Rho(),cluster.position().Eta(),cluster.position().Phi()
        
        self.clusters.append(ROOT.TGraph())
        self.clusters[-1].SetPoint(0,cluster.position().Eta(),cluster.position().Phi())
        self.clusters[-1].SetMarkerStyle(20)
        self.clusters[-1].SetMarkerColor(ROOT.kViolet)

        if links:
            for fraction in cluster.recHitFractions():
                self.clusterLinks.append(ROOT.TGraph())
                self.clusterLinks[-1].SetPoint(0,cluster.position().Eta(),cluster.position().Phi())
                self.clusterLinks[-1].SetPoint(1,fraction.recHitRef().position().Eta(),fraction.recHitRef().position().Phi())
                self.clusterLinks[-1].SetLineColor(ROOT.kViolet)
                


    def addTrack(self,track):
        N = track.nTrajectoryPoints()
        if deltaR(track.trackRef().innerMomentum().Eta(),track.trackRef().innerMomentum().Phi(),self.etaCenter,self.phiCenter)>self.radius:
            return
        self.tracks.append(ROOT.TGraph())
        ii=0
        self.tracks[-1].SetPoint(ii,track.trackRef().innerMomentum().Eta(),track.trackRef().innerMomentum().Phi())
        ii=ii+1
#        for i,point in enumerate(track.trajectoryPoints()):
#            if point.position().Eta() !=0.0 and point.position().Phi()!=0 and i>2:
#                self.tracks[-1].SetPoint(ii,point.position().Eta(),point.position().Phi())
#                ii=ii+1
        for extrap in [4,5,6,7]:        
            pos = track.extrapolatedPoint(extrap).position()
            if pos.Eta() !=0 and pos.Phi() !=0:
                self.tracks[-1].SetPoint(ii,pos.Eta(),pos.Phi())
                ii=ii+1
                
        self.tracks[-1].SetMarkerStyle(7)


    def addSimParticle(self,track):
        if track.charge() !=0:
            return
        p=ROOT.TGraph()
        ii=0

        if abs(track.pdgCode())==22:
                pos = track.extrapolatedPoint(4).position()
                if pos.Eta() !=0 and pos.Phi() !=0 and deltaR(pos.Eta(),pos.Phi(),self.etaCenter,self.phiCenter)<self.radius:
                    p.SetPoint(ii,pos.Eta(),pos.Phi())
                    ii=ii+1
        else:            
                pos = track.extrapolatedPoint(4).position()
                if pos.Eta() !=0 and pos.Phi() !=0 and deltaR(pos.Eta(),pos.Phi(),self.etaCenter,self.phiCenter)<self.radius:
                    p.SetPoint(ii,pos.Eta(),pos.Phi())
                    ii=ii+1

            
        p.SetMarkerStyle(self.markerBySimType(track))
        p.SetMarkerColor(ROOT.kAzure)
        if ii>0:
            self.simParticles.append(p)
        

    def addGenParticle(self,particle):
        if particle.charge()==0:
            return
        if deltaR(particle.eta(),particle.phi(),self.etaCenter,self.phiCenter)>self.radius:
            return

        marker = self.markerByType(particle)
        if marker<0:
            print 'Unknown particle Type',particle.pdgId()
            return
        
        self.genParticles.append(ROOT.TGraph(1))
        self.genParticles[-1].SetPoint(0,particle.eta(),particle.phi())
        self.genParticles[-1].SetMarkerStyle(marker)
        self.genParticles[-1].SetMarkerColor(ROOT.kAzure)
        


    def scaleRecHit(self,hit,fraction):
        newHit = copy.deepcopy(hit)
        corners=[ROOT.TVector2(hit['corners'][0]['eta'],hit['corners'][0]['phi']), \
                 ROOT.TVector2(hit['corners'][1]['eta'],hit['corners'][1]['phi']), \
                 ROOT.TVector2(hit['corners'][2]['eta'],hit['corners'][2]['phi']), \
                 ROOT.TVector2(hit['corners'][3]['eta'],hit['corners'][3]['phi'])]

        centerOfGravity = (corners[0]+corners[1]+corners[2]+corners[3])
        centerOfGravity*=0.25
        radialVectors=[(corners[0]-centerOfGravity),\
                       (corners[1]-centerOfGravity),\
                       (corners[2]-centerOfGravity),\
                       (corners[3]-centerOfGravity)]


        for i in range(0,4):
            radialVectors[i]*=fraction
            newHit['corners'][i]['eta'] = (radialVectors[i]+centerOfGravity).X()
            newHit['corners'][i]['phi'] = (radialVectors[i]+centerOfGravity).Y()
        return newHit    
                       

    def viewEtaPhi(self):
        
        self.etaPhiView = ROOT.TCanvas(self.name+'etaPhiCaNVAS',self.name)
        self.etaPhiView.cd()
        frame=self.etaPhiView.DrawFrame(self.etaCenter-self.radius,self.phiCenter-self.radius,self.etaCenter+self.radius,self.phiCenter+self.radius) 
        frame.GetXaxis().SetTitle("#eta")
        frame.GetYaxis().SetTitle("#phi")
        self.etaPhiView.Draw()
        self.etaPhiView.cd()    
        self.geolines=[]
        self.hitlines=[]
        
        #sort hits by energy
        self.rechits=sorted(self.rechits,key=lambda x: x['energy'],reverse=True)

        #first draw boundaries and calculate fractions at the same time
        
        for hit in self.rechits:
            self.geolines.append(ROOT.TGraph(5))
            self.hitlines.append(ROOT.TGraph(5))
            fraction = hit['energy']/self.rechits[0]['energy']
            for (i,corner) in enumerate(hit['corners']):
                self.geolines[-1].SetPoint(i,corner['eta'],corner['phi'])

            scaledHit = self.scaleRecHit(hit,fraction)
            for (i,corner) in enumerate(scaledHit['corners']):
                self.hitlines[-1].SetPoint(i,corner['eta'],corner['phi'])
            


            self.geolines[-1].SetPoint(4,hit['corners'][0]['eta'],hit['corners'][0]['phi'])
            self.hitlines[-1].SetPoint(4,scaledHit['corners'][0]['eta'],scaledHit['corners'][0]['phi'])

            self.geolines[-1].SetLineColor(ROOT.kGray)
            self.geolines[-1].SetLineStyle(hit['depth'])
            self.geolines[-1].Draw("Lsame")
            if hit['depth'] ==1:
                self.hitlines[-1].SetLineColor(ROOT.kRed)
            elif hit['depth'] ==2:
                self.hitlines[-1].SetLineColor(ROOT.kGreen)
            elif hit['depth'] ==3:
                self.hitlines[-1].SetLineColor(ROOT.kMagenta)
            elif hit['depth'] ==4:
                self.hitlines[-1].SetLineColor(ROOT.kBlack)
            elif hit['depth'] ==5:
                self.hitlines[-1].SetLineColor(ROOT.kYellow)
            self.hitlines[-1].SetLineWidth(2)
            self.hitlines[-1].Draw("Lsame")



        for particle in self.genParticles:
            particle.Draw("Psame")

        for track in self.tracks:
            track.Draw("PLsame")

        for track in self.simParticles:
            track.Draw("PLsame")

        for cluster in self.clusters:
            cluster.Draw("Psame")
        for link in self.clusterLinks:
            link.Draw("Lsame")
            
            


                
        self.etaPhiView.Update()
        
