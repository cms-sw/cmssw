class ParticleData(object):
    """ Class for holding particle data such as charge.
    """
    def __init__(self, charge=0):
        self.charge=charge
        
    def __repr__(self):
        return "charge="+str(self.charge)

class ParticleDataList(object):
    """ Class for generic handling particle ids, names and properties.
    
    Multiple ids can be mapped to multiple names of particle.
    First name/id in the list is the default name. But additional names/ids can be given.
    An examples can be found in the defaultParticleDataList.
    """
    def __init__(self, list=None):
        """ A list of particle ids and names can be given to the constructor.
        """
        self._list = []
        if list != None:
            self._list = list
    
    def setList(self, list):
        self._list = list
    
    def getList(self):
        return self._list
    
    def addParticle(self, ids, names, particleData):
        """ Add a paricle with (multiple) ids and names to the list.
        """
        if not (isinstance(ids,list) and isinstance(names,list)):
            raise TypeError("addParticle needs to lists as input: e.g. [1,-1],['d','dbar']")
        self._list += [(ids, names, particleData)]
    
    def getDefaultName(self, name):
        """ Return the default (first in list) name given any of the particle's names.
        """
        for items in self._list:
            if name in items[1]:
                return items[1][0]
        return name
    
    def getDefaultId(self, id):
        """ Return the default (first in list) id given any of the particle's ids.
        """
        for items in self._list:
            if id in items[0]:
                return items[0][0]
        return id
    
    def getIdFromName(self, name):
        """ Return the default (first in list) id given any of the particle's names.
        """
        for items in self._list:
            if name in items[1]:
                return items[0][0]
        return 0
    
    def getNameFromId(self, id):
        """ Return the default (first in list) name given any of the particle's ids.
        """
        for items in self._list:
            if id in items[0]:
                return items[1][0]
        return "unknown"
    
    def getParticleDataFromId(self, id):
        for items in self._list:
            if id in items[0]:
                return items[2]

    def isQuarkId(self, id):
        return abs(id) in [1, 2, 3, 4, 5, 6]
        
    def isLeptonId(self, id):
        return abs(id) in [11, 12, 13, 14, 15, 16]
        
    def isGluonId(self, id):
        return abs(id) in [21, 9]
        
    def isBosonId(self, id):
        return abs(id) in [21, 9, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37]
        
    def isPhotonId(self, id):
        return id == 22
        
    def isHiggsId(self, id):
        return abs(id) in [25, 35, 36, 37]
    
    def isSusyId(self, id):
        return abs(id) in [1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 2000001, 2000002, 2000003, 2000004, 2000005, 2000006, 2000011, 2000013, 1000021, 1000022, 1000023, 1000024, 1000025, 1000035, 1000037, 1000039]

defaultQuarkDataList = ParticleDataList([
([1, - 1], ["d", "d_quark", "dbar"], ParticleData(1.0/3.0)),
([2, - 2], ["u", "u_quark", "ubar"], ParticleData(2.0/3.0)),
([3, - 3], ["s", "s_quark", "sbar"], ParticleData(1.0/3.0)),
([4, - 4], ["c", "c_quark", "cbar"], ParticleData(2.0/3.0)),
([5, - 5], ["b", "b_quark", "bbar"], ParticleData(1.0/3.0)),
([6, - 6], ["t", "t_quark", "tbar"], ParticleData(2.0/3.0))
])

defaultLeptonDataList = ParticleDataList([
([11, - 11], ["e","electron", "Electron", "e+", "e-"], ParticleData(1)),
([12, - 12], ["nu_e", "Electron_neutrino", "electron_neutrino", "nu_electron"], ParticleData(0)),
([13, - 13], ["mu", "Muon", "muon", "mu+", "mu-"], ParticleData(1)),
([14, - 14], ["nu_mu", "nu_muon", "Muon_neutrino", "muon_neutrino"], ParticleData(0)),
([15, - 15], ["tau", "Tau", "tau+", "tau-"], ParticleData(1)),
([16, - 16], ["nu_tau", "Tau_neutrino", "tau_neutrino"], ParticleData(0))
])

defaultBosonDataList = ParticleDataList([
([21, 9], ["g", "Gluon", "gluon"], ParticleData(0)),
([22], ["gamma", "Photon", "photon"], ParticleData(0)),
([23], ["Z", "Z_boson"], ParticleData(0)),
([24, - 24], ["W", "W_boson", "W+", "W-"], ParticleData(1)),
([25], ["h", "Higgs_boson", "Higgs", "higgs_boson"], ParticleData(0))
])

defaultHadronDataList = ParticleDataList([
([111], ["pi0", "Pi0"], ParticleData(0)),
([112], ["pi+", "Pi+"], ParticleData(1)),
([221], ["eta", "Eta"], ParticleData(0)),
([130], ["K0_L"], ParticleData(0)),
([310], ["K0_S"], ParticleData(0)),
([311], ["K0"], ParticleData(0)),
([321], ["K+"], ParticleData(1)),
([411], ["D0"], ParticleData(0)),
([421], ["D+"], ParticleData(1)),
([511], ["B0"], ParticleData(0)),
([521], ["B+"], ParticleData(1)),
([2212], ["p","Proton","proton"], ParticleData(1)),
([2112], ["n","Neutron","neutron"], ParticleData(0)),
([2224], ["Delta++"], ParticleData(2)),
([2214], ["Delta+"], ParticleData(1)),
([2114], ["Delta0"], ParticleData(0)),
([1114], ["Delta-"], ParticleData(1))
])

defaultExtensionDataList = ParticleDataList([
([32], ["Z'", "Z_prime"], ParticleData(0)),
([33], ["Z''", "Z_primeprime"], ParticleData(0)),
([34, - 34], ["W'", "W_prime", "W'+", "W'-"], ParticleData(1)),
([37, - 37], ["H+", "Charged_Higgs", "H+", "H-"], ParticleData(1)),
([35], ["H0", "Neutral_Higgs_H", "H"], ParticleData(0)),
([36], ["A0", "Neutral_Higgs_A", "A"], ParticleData(0))
])

defaultSusyDataList = ParticleDataList([
([1000001, - 1000001], ["d_squark_L", "d~_L", "d~_L_bar"], ParticleData(1.0/3.0)),
([1000002, - 1000002], ["u_squark_L", "u~_L", "u~_L_bar"], ParticleData(2.0/3.0)),
([1000003, - 1000003], ["s_squark_L", "s~_L", "s~_L_bar"], ParticleData(1.0/3.0)),
([1000004, - 1000004], ["c_squark_L", "c~_L", "c~_L_bar"], ParticleData(2.0/3.0)),
([1000005, - 1000005], ["sbottom_L", "b~_1", "b~_1_bar"], ParticleData(1.0/3.0)),
([1000006, - 1000006], ["stop_L", "t~_1", "t~_1_bar"], ParticleData(2.0/3.0)),

([1000011, - 1000011], ["Selectron_L", "selectron_L", "e~_L", "e~_L+", "e~_L-"], ParticleData(1)),
([1000012, - 1000012], ["Electron_sneutrino", "electron_sneutrino", "nu~_e_L"], ParticleData(0)),
([1000013, - 1000013], ["Smuon_L", "smuon_L", "mu~_L", "mu~_L+", "mu~_L-"], ParticleData(1)),
([1000014, - 1000014], ["Muon_sneutrino", "muon_sneutrino", "nu~_mu_L"], ParticleData(0)),
([1000015, - 1000015], ["Stau_1", "stau_1", "tau~_1+", "tau~_1-"], ParticleData(1)),
([1000016, - 1000016], ["Tau_sneutrino", "tau_sneutrino", "nu~_tau_L"], ParticleData(0)),

([2000001, - 2000001], ["d_squark_R", "d~_L", "d~_L_bar"], ParticleData(1.0/3.0)),
([2000002, - 2000002], ["u_squark_R", "u~_L", "u~_L_bar"], ParticleData(2.0/3.0)),
([2000003, - 2000003], ["s_squark_R", "s~_L", "s~_L_bar"], ParticleData(1.0/3.0)),
([2000004, - 2000004], ["c_squark_R", "c~_L", "c~_L_bar"], ParticleData(2.0/3.0)),
([2000005, - 2000005], ["sbottom_R", "b~_2", "b~_2_bar"], ParticleData(1.0/3.0)),
([2000006, - 2000006], ["stop_R", "t~_2", "t~_2_bar"], ParticleData(2.0/3.0)),

([2000011, - 2000011], ["Selectron_R", "selectron_R", "e~_R", "e~_R+", "e~_R-"], ParticleData(1)),
([1000013, - 1000013], ["Smuon_R", "smuon_R", "mu~_L", "mu~_R+", "mu~_R-"], ParticleData(1)),
([1000015, - 1000015], ["Stau_2", "stau_2", "tau~_2+", "tau~_2 -"], ParticleData(1)),

([1000021], ["Gluino", "gluino", "g~"], ParticleData(0)),
([1000022, - 1000022], ["Neutralino_1", "neutralino_1", "chi~_1"], ParticleData(0)),
([1000023, - 1000023], ["Neutralino_2", "neutralino_2", "chi~_2"], ParticleData(0)),
([1000025, - 1000025], ["Neutralino_3", "neutralino_3", "chi~_3"], ParticleData(0)),
([1000035, - 1000035], ["Neutralino_4", "neutralino4", "chi~_4"], ParticleData(0)),

([1000024, - 1000024], ["Chargino_1", "chargino_1", "chi~_1+", "chi~_1-"], ParticleData(1)),
([1000037, - 1000037], ["Chargino_2", "chargino_2", "chi~_2+", "chi~_2-"], ParticleData(1)),

([1000039], ["Gravitino", "gravitino", "G"], ParticleData(0))
])

defaultParticleDataList = ParticleDataList(
defaultQuarkDataList.getList() + 
defaultLeptonDataList.getList() + 
defaultBosonDataList.getList() + 
defaultHadronDataList.getList() + 
defaultExtensionDataList.getList() + 
defaultSusyDataList.getList())

partonParticleDataList = ParticleDataList([
([1, - 1, 2, - 2, 3, - 3, 4, - 4, 21, 9], ["parton", "d", "dbar", "u", "ubar", "s", "sbar", "c", "cbar", "b", "bbar", "t", "tbar", "gluon", "g"], ParticleData())
] + 
defaultLeptonDataList.getList() + [ 
([22], ["gamma", "Photon", "photon"], ParticleData(0)),
([23], ["Z", "Z_boson"], ParticleData(0)),
([24, - 24], ["W", "W_boson", "W+", "W-"], ParticleData(1)),
([25], ["h", "Higgs_boson", "Higgs", "higgs_boson"], ParticleData(1))
])
