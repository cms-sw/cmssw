from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from ROOT import TFile, TH1F
from ROOT import TVector3, TLorentzVector
from PhysicsTools.HeppyCore.papas.path import Helix
import math
from PhysicsTools.HeppyCore.utils.deltar import deltaR

class ImpactParameter(Analyzer):
    '''This analyzer puts an impact parameter for every charged particle
    as an attribute of its path.
    The significance is calculated, the calculus are a first order approximation,
    thus this may not be correct for large impact parameters (more than 3 mm).
    The Impact parameter significance is stored in the particle's path.
    
    New attributes for PhysicsTools.HeppyCore.papas.pfobjects.Particle.path (from compute_IP) :
    *   closest_t = time of closest approach to the primary vertex.
    *   IP = signed impact parameter
    *   IPcoord = TVector3 of the point of closest approach to the
        primary vertex
        
    New attributes for particles.path (from compute_theta_0) :
    *   theta_0 = 1/sqrt(2) * gaussian width of the scattering angle
        due to the beam pipe. -->  See pdg booklet, Passage of particles through matter,
        multiple scattering through small angles.
    *   xX_0 = distance in number of X_0 the radiation length the particles goes through
    
    New attributes for particles.path (from compute_IP_signif) :
    *   IP_signif = the impact parameter significance. To get the uncertainty, just compute IP/IP_signif
    *   IP_sigma = the uncertainty on the impact parameter
        
    Then, several b-tagging methods are applied to each jet, with selected tracks :
    *   a log-likelihood ratio based on Impact Parameter (IP_b_LL),
        if a numerator and a denominator for IP are provided (see example)
    *   a log-likelihood ratio based on Impact Parameter Significance (IPs_b_LL),
        if a numerator and a denominator for IPs are provided (see example)
    *   TCHE and TCHP taggers : using the second and third highest Impact Prameter for each jet
    
    New attributes for jets :
    *   TCHE = the value of the TCHE for this jet
    *   TCHP = the value of the TCHP for this jet
    *   TCHE_IP = the value of the IP of the track used to compute the TCHE
    *   TCHP_IP = the value of the IP of the track used to compute the TCHE
    *   TCHE_x = the x position of the vertex for the particle used for the TCHE
    *   TCHE_y = the y position of the vertex for the particle used for the TCHE
    *   TCHE_z = the z position of the vertex for the particle used for the TCHE
    *   TCHE_pt = the transverse impulsion of the particle used for the TCHE
    *   TCHE_dr = the cone containing the track used for the THCE with respects to the jet axis
    
    Example of configuration : the root files contain the distributions of IP or IPs histograms (h_u, h_b ...)
    that can be divided (num/denom) to get the ratio.

    from PhysicsTools.HeppyCore.papas.detectors.CMS import CMS
    from PhysicsTools.HeppyCore.analyzers.ImpactParameter import ImpactParameter
    btag = cfg.Analyzer(
        ImpactParameter,
        jets = 'jets',
        
        # needed only for the IP_b_LL tagger
        # file and histogram key for b jet charged hadrons IP
        # num_IP = ("histo_stat_IP_ratio_bems.root","h_b"),
        # file and histogram key for u jet charged hadrons IP
        # denom_IP = ("histo_stat_IP_ratio_bems.root","h_u"),
        
        # needed only for the IPs_b_LL tagger
        # file and histogram key for b jet charged hadrons IPs
        # num_IPs = ("histo_stat_IPs_ratio_bems.root","h_b"),
        # file and histogram key for u jet charged hadrons IPs
        # denom_IPs = ("histo_stat_IPs_ratio_bems.root","h_u"),

        # selection of charged hadrons for b tagging 
        pt_min = 1, # pt threshold 
        dxy_max = 2e-3, # max impact parameter in transverse plane, in m, w/r origin 
        dz_max = 17e-2, # max longitudinal impact parameter in transverse plane, in m, w/r origin
        detector = CMS()
    )    
    
    '''
    def beginLoop(self, setup):
        super(ImpactParameter, self).beginLoop(setup)
        if hasattr(self.cfg_ana, 'num_IP') == False :
            self.tag_IP_b_LL = False
        else :
            if hasattr(self.cfg_ana, 'denom_IP') == False :
                self.tag_IP_b_LL = False
                raise AttributeError('You specified a numerator without a denominator for the log likelihood based on IP')
            else :
                self.tag_IP_b_LL = True
                self.num_IP_file = TFile.Open(self.cfg_ana.num_IP[0])
                self.num_IP_hist = self.num_IP_file.Get(self.cfg_ana.num_IP[1])
                self.denom_IP_file = TFile.Open(self.cfg_ana.denom_IP[0])
                self.denom_IP_hist = self.denom_IP_file.Get(self.cfg_ana.denom_IP[1])
                self.ratio_IP = TH1F("ratio_IP","num_IP over denom_IP", self.num_IP_hist.GetXaxis().GetNbins(), self.num_IP_hist.GetXaxis().GetXmin(), self.num_IP_hist.GetXaxis().GetXmax())
                self.ratio_IP.Divide(self.num_IP_hist,self.denom_IP_hist)
                #import pdb; pdb.set_trace()
                
        if hasattr(self.cfg_ana, 'num_IPs') == False :
            self.tag_IPs_b_LL = False
        else :
            if hasattr(self.cfg_ana, 'denom_IPs') == False :
                self.tag_IPs_b_LL = False
                raise AttributeError('You specified a numerator without a denominator for the log likelihood based on IP significance')
            else :
                self.tag_IPs_b_LL = True
                self.num_IPs_file = TFile.Open(self.cfg_ana.num_IPs[0])
                self.num_IPs_hist = self.num_IPs_file.Get(self.cfg_ana.num_IPs[1])
                self.denom_IPs_file = TFile.Open(self.cfg_ana.denom_IPs[0])
                self.denom_IPs_hist = self.denom_IPs_file.Get(self.cfg_ana.denom_IPs[1])
                self.ratio_IPs = TH1F("ratio_IPs","num_IPs over denom_IPs", self.num_IPs_hist.GetXaxis().GetNbins(), self.num_IPs_hist.GetXaxis().GetXmin(), self.num_IPs_hist.GetXaxis().GetXmax())
                self.ratio_IPs.Divide(self.num_IPs_hist,self.denom_IPs_hist)


    def ll_tag(self, ratio_histo, ptc_var, jet_tag ):
        ibin = ratio_histo.FindBin(ptc_var)
        lhratio = ratio_histo.GetBinContent(ibin)
        if not lhratio == 0:
            LLratio = math.log(lhratio)
            jet_tag += LLratio
        if lhratio == 0:
            LLratio = 0
        return LLratio


    def process(self, event):
        assumed_vertex = TVector3(0, 0, 0)
        jets = getattr(event, self.cfg_ana.jets)
        detector = self.cfg_ana.detector
        pt_min = self.cfg_ana.pt_min
        dxy_max = self.cfg_ana.dxy_max
        dz_max = self.cfg_ana.dz_max
        for jet in jets:
            IP_b_LL = 0     # value of the log likelihood ratio based on IP initiated at 0
            IPs_b_LL = 0    # value of the log likelihood ratio based on IP significance initiated at 0
            ipsig_ptcs = [] # list of IP signif and associated ptcs
            for id, ptcs in jet.constituents.iteritems():
                if abs(id) in [22,130,11]:
                    continue
                for ptc in ptcs :
                    if ptc.q() == 0 :
                        continue
                    ptc.path.compute_IP(assumed_vertex,jet)
                    
                    ptc_IP_signif = 0
                    if hasattr(ptc.path, 'points') == True and 'beampipe_in' in ptc.path.points:
                        phi_in = ptc.path.phi(ptc.path.points['beampipe_in'].X(),\
                                                    ptc.path.points['beampipe_in'].Y())
                        phi_out= ptc.path.phi(ptc.path.points['beampipe_out'].X(),\
                                                    ptc.path.points['beampipe_out'].Y())
                        deltat = ptc.path.time_at_phi(phi_out)-ptc.path.time_at_phi(phi_in)
                        x = ptc.path.path_length(deltat)
                        X_0 = detector.elements['beampipe'].material.x0
                        ptc.path.compute_theta_0(x, X_0)
                        ptc.path.compute_IP_signif(ptc.path.IP,
                                                   ptc.path.theta_0,
                                                   ptc.path.points['beampipe_in'])
                    else :
                        ptc.path.compute_IP_signif(ptc.path.IP, None, None)
                    
                    dx = ptc.path.IPcoord.x() - assumed_vertex.x()
                    dy = ptc.path.IPcoord.y() - assumed_vertex.y()
                    dz = ptc.path.IPcoord.z() - assumed_vertex.z()
                    if ptc.path.p4.Perp() > pt_min and (dx**2 + dy**2)**0.5 < dxy_max and dz**2 < dz_max**2 :
                        ipsig_ptcs.append([ptc.path.IP_signif, ptc])
                        
                        if self.tag_IP_b_LL:
                            ptc.path.IP_b_LL = self.ll_tag(self.ratio_IP, ptc.path.IP,IP_b_LL )
                        if self.tag_IPs_b_LL:
                            ptc.path.IPs_b_LL = self.ll_tag(self.ratio_IPs, ptc.path.IP_signif, IPs_b_LL )
                            
            ipsig_ptcs.sort(reverse=True)
            
            if len(ipsig_ptcs) < 2 :
                TCHE = -99
                TCHP = -99
                TCHE_IP = -99
                TCHP_IP = -99
                TCHE_x = -99
                TCHE_y = -99
                TCHE_z = -99
                TCHE_pt = -99
                TCHE_dr = -99
                
            if len(ipsig_ptcs) > 1 :
                TCHE = ipsig_ptcs[1][0]
                ptc = ipsig_ptcs[1][1]
                TCHE_IP = ptc.path.IP
                TCHE_x, TCHE_y, TCHE_z = ptc.path.coord_at_time(0)
                TCHE_pt = ptc.path.p4.Perp()
                TCHE_dr = deltaR(jet.eta(), jet.phi(), ptc.eta(), ptc.phi())
                TCHP = -99
                TCHP_IP = -99
                
            if len(ipsig_ptcs) > 2 :
                TCHP = ipsig_ptcs[2][0]
                ptc = ipsig_ptcs[2][1]
                TCHP_IP = ptc.path.IP
                
            jet.tags['IP_b_LL'] = IP_b_LL  if self.tag_IP_b_LL  else None
            jet.tags['IPs_b_LL']= IPs_b_LL if self.tag_IPs_b_LL else None
            #TODO COLIN : create a BTagInfo class. 
            jet.tags['TCHE'] = TCHE
            jet.tags['TCHP'] = TCHP
            jet.tags['TCHE_IP'] = TCHE_IP
            jet.tags['TCHP_IP'] = TCHP_IP
            jet.tags['TCHE_x'] = TCHE_x
            jet.tags['TCHE_y'] = TCHE_y
            jet.tags['TCHE_z'] = TCHE_z
            jet.tags['TCHE_xy'] = (TCHE_x**2+TCHE_y**2)**0.5
            jet.tags['TCHE_pt'] = TCHE_pt
            jet.tags['TCHE_dr'] = TCHE_dr
            
            if hasattr(event, 'K0s') == True :
                jet.tags['K0s'] = event.K0s
            else :
                jet.tags['K0s'] = -99
            if hasattr(event, 'Kp') == True :
                jet.tags['Kp'] = event.Kp
            else :
                jet.tags['Kp'] = -99
            if hasattr(event, 'L0') == True :
                jet.tags['L0'] = event.L0
            else :
                jet.tags['L0'] = -99
            if hasattr(event, 'S0') == True :
                jet.tags['S0'] = event.S0
            else :
                jet.tags['S0'] = -99
            if hasattr(event, 'Sp') == True :
                jet.tags['Sp'] = event.Sp
            else :
                jet.tags['Sp'] = -99
            if hasattr(event, 'Sm') == True :
                jet.tags['Sm'] = event.Sm
            else :
                jet.tags['Sm'] = -99
            if hasattr(event, 'Muons') == True :
                jet.tags['Muons'] = event.Muons
            else :
                jet.tags['Muons'] = -99
        

