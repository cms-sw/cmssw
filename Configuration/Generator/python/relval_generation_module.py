######################################################
#                                                    #
#       relval_simulation_module                     #
#                                                    #  
#  This module is a collection of the simulation     # 
#  procedues. The random number service is built     #
#  by the function random_generator_service(energy)  #
#                                                    #
######################################################

import FWCore.ParameterSet.Config as cms
import relval_common_module as common

from math import pi as PI
import os
import sys

#---------------------------------------------------
# This just simplifies the use of the logger
mod_id="["+os.path.basename(sys._getframe().f_code.co_filename)[:-3]+"]"

#----------------------------
# Some useful constants:
ETA_MAX=2.5
ETA_MIN=-2.5

def generate(step, evt_type, energy, evtnumber):
    """
    This function calls all the other functions specific for
    an event evt_type.
    """
   
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")
    
    # Build the switch cases:
    
    # Particle Gun
    if evt_type in ("MU+","MU-","E","DIE","GAMMA","TAU","PI0","PI+","PI-"):
       generator = _generate_PGUN\
         (step, evt_type, energy, evtnumber)
     
    elif evt_type in ("HZZMUMUMUMU", "HZZEEEE", "HZZTTTT", "HZZLLLL","HGG"):
       generator = _generate_Higgs\
         (step, evt_type, energy, evtnumber)
     
    elif evt_type in ("B_JETS", "C_JETS"):
       generator = _generate_udscb_jets\
         (step, evt_type, energy, evtnumber)        
    
    elif evt_type in ("QCD","TTBAR","ZPJJ","MINBIAS","RS1GG","HpT"):
        generator = eval("_generate_"+evt_type+"(step, evt_type, energy, evtnumber)") 
    
    elif evt_type in ("ZEE","ZTT","ZMUMU"):
        generator = _generate_Zll\
         (step, evt_type, energy, evtnumber)

    elif evt_type in ("ZPEE","ZPTT","ZPMUMU"):
        generator = _generate_ZPll\
         (step, evt_type, energy, evtnumber)         
             
    elif evt_type in ("WE","WM","WT"):
        generator = _generate_Wl(step, evt_type, energy, evtnumber)
         
    else:
      raise "Event type","Type not yet implemented."
             
    common.log( func_id+" Returning Generator")
    
    return generator

#------------------------------       

def _generate_PGUN(step, evt_type, energy, evtnumber):
    """
    Here the settings for the simple generation of a muon, electron or gamma
    are stated.
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")

   # pythia ID: configure the ID of the particle through a dictionary
    py_id_dict = {"MU-":13, 
                  "MU+":-13,
                  "E" :11,
                  "DIE":11,
                  "TAU":15,
                  "GAMMA":22,
                  "PI+":211,
                  "PI-":-211,
                  "PI0":111}
    
    # Build the id string of the event name:
    id_string = evt_type+" "+energy+" nevts "+ str(evtnumber)    
                  
    # We have to check if we want to generate a particle with pt=X or Energy=X                  
    pt_flag=True
    if 'pt' in energy[0:2] or \
       'Pt' in energy[0:2] or \
       'PT' in energy[0:2]:
        energy=energy[2:]
    else:
        pt_flag=False         
                  
    # Energy boundaries are now set:      
    lower_energy = ""
    upper_energy = ""
    


    # Build the partID string
    part_id = cms.untracked.vint32 ()

    part_id.append(py_id_dict[evt_type])
    upper_energy=''
    lower_energy=''
    if energy.find('_')!=-1:
        upper_energy,lower_energy=energy_split(energy)
    else:    
        epsilon= 0.001
        lower_energy = str ( int(energy) - epsilon) # We want a calculation and
        upper_energy = str ( int(energy) + epsilon) # the result as a string   
    
    # Build the process source
    if evt_type in ("TAU","E"):
        # Add the corresponding opposite sign particle. Only for taus and es.
        part_id.append(-1*part_id[0])
    
    antip_flag=False
    if evt_type=="DIE":
        antip_flag=True  
    
    if pt_flag:        
        common.log( func_id+ "This is a pt particle gun ..." )
        generator = cms.EDProducer("FlatRandomPtGunProducer",
                            psethack = cms.string(id_string),
                            firstRun = cms.untracked.uint32(1),
                            PGunParameters = cms.PSet(
                                PartID = part_id,
                                MinEta = cms.double(ETA_MAX),
                                MaxEta = cms.double(ETA_MIN),
                                MinPhi = cms.double(-PI),
                                MaxPhi = cms.double(PI),
                                MinPt  = cms.double(lower_energy),
                                MaxPt  = cms.double(upper_energy) 
                            ),
                            AddAntiParticle=cms.bool(antip_flag),
                            Verbosity = cms.untracked.int32(0)
                        )
    else:
        common.log( func_id+ " This is an Energy particle gun ..." )
        generator = cms.EDProducer("FlatRandomEGunProducer",
                            psethack = cms.string(id_string),
                            firstRun = cms.untracked.uint32(1),
                            PGunParameters = cms.PSet(
                                PartID = part_id,
                                MinEta = cms.double(ETA_MAX),
                                MaxEta = cms.double(ETA_MIN),
                                MinPhi = cms.double(-PI),
                                MaxPhi = cms.double(PI),
                                MinE = cms.double(lower_energy),
                                MaxE = cms.double(upper_energy) 
                            ),
                            AddAntiParticle=cms.bool(antip_flag),
                            Verbosity = cms.untracked.int32(0)
                        )       
                        
    common.log( func_id+" Returning Generator...")
        
    return generator 
   
#---------------------------
    
def _generate_QCD(step, evt_type, energy, evtnumber):
    """
    Here the settings for the generation of QCD events 
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")   
        
    # Recover the energies from the string:
    upper_energy, lower_energy = energy_split(energy)
    
    # Build the process source   
    generator = cms.EDFilter("Pythia6GeneratorFilter",
                        pythiaPylistVerbosity=cms.untracked.int32(0),
                        pythiaHepMCVerbosity=cms.untracked.bool(False),
                        maxEventsToPrint = cms.untracked.int32(0), 
                        filterEfficiency = cms.untracked.double(1),  
                        PythiaParameters = cms.PSet\
                        (parameterSets = cms.vstring\
                                            ("pythiaUESettings",
                                            "processParameters"),
                            pythiaUESettings = user_pythia_ue_settings(),
                            processParameters = cms.vstring("MSEL=1",
                                                "CKIN(3)="+upper_energy,
                                                "CKIN(4)="+lower_energy))
                        )
     
    common.log( func_id+" Returning Generator...")                 
    return generator
 
#---------------------------------

def _generate_MINBIAS(step, evt_type, energy, evtnumber):
    """
    Settings for MINBIAS events generation
    """
    
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")     
    
    # Build the process source   
    generator = cms.EDFilter("Pythia6GeneratorFilter",
                      pythiaPylistVerbosity=cms.untracked.int32(0),
                      pythiaHepMCVerbosity=cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0), 
                      filterEfficiency = cms.untracked.double(1),  
                        PythiaParameters = cms.PSet\
                        (parameterSets = cms.vstring\
                                            ("pythiaUESettings",
                                             "processParameters"),
                            pythiaUESettings = user_pythia_ue_settings(),
                            processParameters = cms.vstring(
                                                "MSEL=0",
                                                "MSUB(11)=1",
                                                "MSUB(12)=1",
                                                "MSUB(13)=1",
                                                "MSUB(28)=1",
                                                "MSUB(53)=1",
                                                "MSUB(68)=1",
                                                "MSUB(92)=1",
                                                "MSUB(93)=1",
                                                "MSUB(94)=1",
                                                "MSUB(95)=1"))
                        )
    common.log( func_id+" Returning Generator...")                 
    
    return generator   
    
#---------------------------------

def _generate_Higgs(step, evt_type, energy, evtnumber):
    """    
    Here the settings for the generation of Higgs->ZZ->ll events 
    The energy parameter is not used. According to the evt_type ("HZZMUMUMUMU" 
    or "HZZEEEE") the final state is chosen.
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")      
    
    # Choose between muon, tau or electron decay of the Z
    z_flag="0"
    electron_flag = "0"
    muon_flag = "0"
    tau_flag = "0"
    gamma_flag="0"
    if evt_type == "HZZEEEE":
        electron_flag = "1"
        z_flag="1"
    elif evt_type == "HZZMUMUMUMU":
        muon_flag = "1"    
        z_flag="1"
    elif evt_type == "HZZTTTT":
        tau_flag = "1"
        z_flag="1"
    elif evt_type == "HZZLLLL":
        electron_flag=muon_flag=tau_flag= "1"
        z_flag="1"
    elif evt_type == "HGG":
        gamma_flag="1"    

  
    # Prepare The Pythia params  
    params = cms.vstring(
        "PMAS(25,1)=%s" %energy,      #mass of Higgs",
        "MSEL=0",                  
        #(D=1) to select between full user control
        #(0, then use MSUB) and some preprogrammed alternative: QCD hight pT
        #processes (1, then ISUB=11, 12, 13, 28, 53, 68), QCD low pT processes
        #(2, then ISUB=11, #12, 13, 28, 53, 68, 91, 92, 94, 95)",
        #
	#Check on possible errors during program
        #execution",
        "MSUB(102)=1",             #ggH",
        "MSUB(123)=1",             #ZZ fusion to H",
        "MSUB(124)=1",             #WW fusion to H",
        "CKIN(45)=5.",                       
        "CKIN(46)=150.",           
        #high mass cut on secondary resonance m1 in
        #2->1->2 process Registered by Alexandre.Nikitenko@cern.ch",
        "CKIN(47)=5.",             
        #low mass cut on secondary resonance m2 in
        #2->1->2 process Registered by Alexandre.Nikitenko@cern.ch",
        "CKIN(48)=150.",           
        #high mass cut on secondary resonance m2 in
        #2->1->2 process Registered by Alexandre.Nikitenko@cern.ch",
        "MDME(174,1)=0",           #Z decay into d dbar",        
        "MDME(175,1)=0",          #Z decay into u ubar",
        "MDME(176,1)=0",           #Z decay into s sbar",
        "MDME(177,1)=0",           #Z decay into c cbar",
        "MDME(178,1)=0",           #Z decay into b bbar",
        "MDME(179,1)=0",           #Z decay into t tbar",
        "MDME(182,1)=%s" %electron_flag,#Z decay into e- e+",
        "MDME(183,1)=0",           #Z decay into nu_e nu_ebar",
        "MDME(184,1)=%s" %muon_flag,#Z decay into mu- mu+",
        "MDME(185,1)=0",           #Z decay into nu_mu nu_mubar",
        "MDME(186,1)=%s" %tau_flag,#Z decay into tau- tau+",
        "MDME(187,1)=0",          #Z decay into nu_tau nu_taubar",
        "MDME(210,1)=0",           #Higgs decay into dd",
        "MDME(211,1)=0",           #Higgs decay into uu",
        "MDME(212,1)=0",           #Higgs decay into ss",
        "MDME(213,1)=0",           #Higgs decay into cc",
        "MDME(214,1)=0",           #Higgs decay into bb",
        "MDME(215,1)=0",           #Higgs decay into tt",
        "MDME(216,1)=0",           #Higgs decay into",
        "MDME(217,1)=0",           #Higgs decay into Higgs decay",
        "MDME(218,1)=0",           #Higgs decay into e nu e",
        "MDME(219,1)=0",           #Higgs decay into mu nu mu",
        "MDME(220,1)=0",           #Higgs decay into tau nu tau",
        "MDME(221,1)=0",           #Higgs decay into Higgs decay",
        "MDME(222,1)=0",           #Higgs decay into g g",
        "MDME(223,1)=%s" %gamma_flag,#Higgs decay into gam gam",
        "MDME(224,1)=0",           #Higgs decay into gam Z",
        "MDME(225,1)=%s" %z_flag,  #Higgs decay into Z Z",
        "MDME(226,1)=0",           #Higgs decay into W W"
        ) 

    # Build the process source   
    generator = cms.EDFilter('Pythia6GeneratorFilter',
                      pythiaPylistVerbosity=cms.untracked.int32(0),
                      pythiaHepMCVerbosity=cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0), 
                      filterEfficiency = cms.untracked.double(1),  
                      pythiaVerbosity =cms.untracked.bool(False),
                      PythiaParameters = cms.PSet\
                       (parameterSets = cms.vstring('PythiaUESettings','processParameters'),
                        PythiaUESettings = user_pythia_ue_settings(),
                        processParameters=params
                       )     
                     )

    common.log( func_id+" Returning Generator...")
     
    return generator      

#---------------------------------

def _generate_udscb_jets\
        (step, evt_type, energy, evtnumber):
    """
    Here the settings necessary to udscb jets generation are added. According
    to the flavour the Pythia parameters are changed slightly.
    For the time being the energy parameter is not used.
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")
    
    # Recover the energies from the string:
    upper_energy, lower_energy = energy_split(energy)
   
    # According to the evt_type build the Pythia settings:
    pythia_jet_settings=cms.vstring("MSEL=0")  # User defined process
    pythia_jet_settings+=cms.vstring("MSUB(81)=1", #qq->QQ massive
                                     "MSUB(82)=1") #gg->QQ massive
    if evt_type == "C_JETS":
            pythia_jet_settings+=cms.vstring("MSTP(7)=4") #ccbar
            common.log( func_id+" Including settings for c jets")
    else:
            pythia_jet_settings+=cms.vstring("MSTP(7)=5") #bbbar
            common.log( func_id+" Including settings for b jets")
             
    # Common part to all cases         
    pythia_common=cms.vstring("CKIN(3)="+upper_energy,  # Pt low cut 
                              "CKIN(4)="+lower_energy,  # Pt high cut
                              "CKIN(13)=0.",            # eta min            
                              "CKIN(14)=2.5",           # etamax           
                              "CKIN(15)=-2.5",          # -etamin 
                              "CKIN(16)=0"              # -etamax
                              )
    
    pythia_jet_settings+=pythia_common
    
    # Build the process source
    generator = cms.EDFilter('Pythia6GeneratorFilter',
                      pythiaVerbosity =cms.untracked.bool(True),
                      PythiaParameters = cms.PSet\
                               (parameterSets = cms.vstring\
                                                   ("pythiaUESettings","pythiaJets"),
                                pythiaUESettings = user_pythia_ue_settings(),
                                pythiaJets = pythia_jet_settings
                               )
                     )                       
   
    common.log(func_id+" Returning Generator...")
     
    return generator

#-----------------------------------
    
def _generate_TTBAR(step, evt_type, energy, evtnumber):
    """
    Here the settings for the ttbar pairs are added to the process.
    """
      
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log(func_id+" Entering... ")      
    
    # Build the process source    
    generator = cms.EDFilter('Pythia6GeneratorFilter',
                      pythiaPylistVerbosity=cms.untracked.int32(0),
                      pythiaHepMCVerbosity=cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0), 
                      filterEfficiency = cms.untracked.double(1),                     
                      PythiaParameters = cms.PSet\
                                (parameterSets = cms.vstring\
                                                   ('pythiaUESettings',
                                                    'processParameters'),
                                pythiaUESettings = user_pythia_ue_settings(),
                                # Tau jets (config by A. Nikitenko)
                                # No tau -> electron
                                # No tau -> muon
                                processParameters =cms.vstring\
                                    ("MSEL=0",       # userdef process
                                     "MSUB(81)=1",   # qqbar->QQbar
                                     "MSUB(82)=1",   # gg to QQbar
                                     "MSTP(7)=6",    # flavour top
                                     "PMAS(6,1)=175" # top mass
                                     )
                                ) 
                      )  

    common.log(func_id+" Returning Generator...")
     
    return generator   
 
#---------------------------------

def _generate_Zll(step, evt_type, energy, evtnumber):
    """
    Here the settings for the Z ee simulation are added to the process.
    Energy parameter is not used.
    """
      
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")      

    # Choose between muon or electron decay of the Z
    user_param_sets = "pythiaZll"
    electron_flag = "0"
    muon_flag = "0"
    tau_flag = "0"
    if evt_type == "ZEE":
        electron_flag = "1"
    elif evt_type == "ZMUMU":
        muon_flag = "1"    
    elif evt_type == "ZTT":
        tau_flag = "1"
    else:
        electron_flag=muon_flag=tau_flag= "1"    
    
    pythia_param_sets = cms.vstring(
                 "MSEL = 11 ",           
                 "MDME( 174,1) = 0",            #Z decay into d dbar",
                 "MDME( 175,1) = 0",            #Z decay into u ubar",
                 "MDME( 176,1) = 0",            #Z decay into s sbar",
                 "MDME( 177,1) = 0",            #Z decay into c cbar",
                 "MDME( 178,1) = 0",            #Z decay into b bbar",
                 "MDME( 179,1) = 0",            #Z decay into t tbar",
                 "MDME( 182,1) = %s" %electron_flag,#Z decay into e- e+",
                 "MDME( 183,1) = 0",            #Z decay into nu_e nu_ebar",
                 "MDME( 184,1) = %s" %muon_flag,#Z decay into mu- mu+",
                 "MDME( 185,1) = 0",            #Z decay into nu_mu nu_mubar",
                 "MDME( 186,1) = %s" %tau_flag, #Z decay into tau- tau+",
                 "MDME( 187,1) = 0",            #Z decay into nu_tau nu_taubar",
                 "MSTJ( 11) = 3",    #Choice of the fragmentation function",
                 "MSTP( 2) = 1",            #which order running alphaS",
                 "MSTP( 33) = 0",            #(D=0) ",
                 "MSTP( 51) = 7",            #structure function chosen",
                 "MSTP( 81) = 1",            #multiple parton interactions 1 is
                                             #Pythia default,
                 "MSTP( 82) = 4",            #Defines the multi-parton model",
                 "PARJ( 71) = 10.",            #for which ctau  10 mm",
                 "PARP( 82) = 1.9",   #pt cutoff for multiparton interactions",
                 "PARP( 89) = 1000.", #sqrts for which PARP82 is set",
                 "PARP( 83) = 0.5", #Multiple interactions: matter distrbn
                                    #parameter Registered byChris.Seez@cern.ch
                 "PARP( 84) = 0.4",   #Multiple interactions: matterdistribution
                                  #parameter Registered by Chris.Seez@cern.ch
                 "PARP( 90) = 0.16",  #Multiple interactions:rescaling power
                                      #Registered by Chris.Seez@cern.ch
                 "CKIN( 1) = 40.",            #(D=2. GeV)
                 "CKIN( 2) = -1.",            #(D=-1. GeV)      \
                 )     
                 
    # Build the process source
    generator = cms.EDFilter('Pythia6GeneratorFilter', 
                      pythiaPylistVerbosity=cms.untracked.int32(0),
                      pythiaHepMCVerbosity=cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0), 
                      filterEfficiency = cms.untracked.double(1),    
                      PythiaParameters = cms.PSet\
                               (parameterSets = cms.vstring('PythiaUESettings','processParameters'),
                                PythiaUESettings=user_pythia_ue_settings(),
                                processParameters=pythia_param_sets )
                     )

    common.log(func_id+" Returning Generator...")
     
    return generator   
#---------------------------------

def _generate_Wl(step, evt_type, energy, evtnumber):
    """
    Here the settings for the Z ee simulation are added to the process.
    Energy parameter is not used.
    """
      
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")      

    # Choose between muon or electron decay of the Z
    electron_flag = "0"
    muon_flag = "0"
    tau_flag = "0"
    if evt_type == "WE":
        electron_flag = "1"
    elif evt_type == "WM":
        muon_flag = "1"    
    elif evt_type == "WT":
        tau_flag = "1"    
        
    # Build the process source
    generator = cms.EDFilter('Pythia6GeneratorFilter', 
                      pythiaPylistVerbosity=cms.untracked.int32(0),
                      pythiaHepMCVerbosity=cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0), 
                      filterEfficiency = cms.untracked.double(1),
                      PythiaParameters = cms.PSet\
                               (parameterSets = cms.vstring('PythiaUESettings','processParameters'),
                                PythiaUESettings=user_pythia_ue_settings(),
                                processParameters=cms.vstring('MSEL=0    !User defined processes',
                                                              'MSUB(2)     = 1',#    !W production 
                                                              'MDME(190,1) = 0',#    !W decay into dbar u 
                                                              'MDME(191,1) = 0',#    !W decay into dbar c 
                                                              'MDME(192,1) = 0',#    !W decay into dbar t 
                                                              'MDME(194,1) = 0',#    !W decay into sbar u 
                                                              'MDME(195,1) = 0',#    !W decay into sbar c 
                                                              'MDME(196,1) = 0',#    !W decay into sbar t 
                                                              'MDME(198,1) = 0',#    !W decay into bbar u 
                                                              'MDME(199,1) = 0',#    !W decay into bbar c 
                                                              'MDME(200,1) = 0',#    !W decay into bbar t 
                                                              'MDME(205,1) = 0',#    !W decay into bbar tp 
                                                              'MDME(206,1) = %s' %electron_flag,#   !W decay into e+ nu_e 
                                                              'MDME(207,1) = %s' %muon_flag,#   !W decay into mu+ nu_mu 
                                                              'MDME(208,1) = %s' %tau_flag,#   !W decay into tau+ nu_tau
                                                             )
                              )
                     )

    common.log(func_id+" Returning Generator...")
     
    return generator       
                                   
#---------------------------------

def _generate_ZPJJ(step, evt_type, energy, evtnumber):
    """
    Here the settings for the Zprime to JJ simulation are added to the
    process. 
    """
    
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log(func_id+" Entering... ")            
    common.log( func_id+" Returning Generator...")
    # You might wonder why this time it's not pythonised..Me too: due to the excessive fragmentation of the 
    # cfgs it's not worth to do that at the moment. It also obliges to have two functions for the ZP instead of one.
    return common.include_files('Configuration/JetMET/data/calorimetry-gen-Zprime_Dijets_700.cff')[0].source                                   

#---------------------------------

def _generate_ZPll(step, evt_type, energy, evtnumber):
    """
    Here the settings for the Z ee simulation are added to the process.
    Energy parameter is not used.
    """
      
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")      

    # Choose between muon or electron decay of the Z
    electron_flag = "0"
    muon_flag = "0"
    tau_flag = "0"
    if evt_type == "ZPEE":
        electron_flag = "1"
    elif evt_type == "ZPMUMU":
        muon_flag = "1"    
    elif evt_type == "ZPTT":
        tau_flag = "1"
    else:
        electron_flag=muon_flag=tau_flag= "1"    
    
    # Build the process source
    generator = cms.EDFilter('Pythia6GeneratorFilter', 
                      pythiaPylistVerbosity=cms.untracked.int32(0),
                      pythiaHepMCVerbosity=cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0), 
                      filterEfficiency = cms.untracked.double(1),    
                      PythiaParameters = cms.PSet\
                               (parameterSets = cms.vstring('PythiaUESettings','processParameters'),
                                PythiaUESettings=user_pythia_ue_settings(),
                                processParameters=\
                                    cms.vstring('MSEL       = 0    ', 
                                                'MSUB(141)  = 1    ',#  !ff  gamma z0 Z0', 
                                                'MSTP(44)   = 3    ',#  !only select the Z process', 
                                                'PMAS(32,1) = %s' %energy,#  !mass of Zprime', 
                                                'CKIN(1)    = 400  ',#  !(D=2. GeV)', 
                                                'MDME(289,1)= 0    ',#  !d dbar', 
                                                'MDME(290,1)= 0    ',#  !u ubar', 
                                                'MDME(291,1)= 0    ',#  !s sbar', 
                                                'MDME(292,1)= 0    ',#  !c cbar', 
                                                'MDME(293,1)= 0    ',#  !b bar', 
                                                'MDME(294,1)= 0    ',#  !t tbar', 
                                                'MDME(295,1)= 0    ',#  !4th gen Q Qbar', 
                                                'MDME(296,1)= 0    ',# !4th gen Q Qbar', 
                                                'MDME(297,1)= %s ' %electron_flag,#  !e e', 
                                                'MDME(298,1)= 0    ',#  !neutrino e e', 
                                                'MDME(299,1)= %s ' %muon_flag,#  ! mu mu', 
                                                'MDME(300,1)= 0    ',#  !neutrino mu mu', 
                                                'MDME(301,1)= %s    ' %tau_flag,#  !tau tau', 
                                                'MDME(302,1)= 0    ',#  !neutrino tau tau', 
                                                'MDME(303,1)= 0    ',#  !4th generation lepton', 
                                                'MDME(304,1)= 0    ',#  !4th generation neutrino', 
                                                'MDME(305,1)= 0    ',#  !W W', 
                                                'MDME(306,1)= 0    ',#  !H  charged higgs', 
                                                'MDME(307,1)= 0    ',#  !Z', 
                                                'MDME(308,1)= 0    ',#  !Z', 
                                                'MDME(309,1)= 0    ',#  !sm higgs', 
                                                'MDME(310,1)= 0    ' #  !weird neutral higgs HA')
                                               )
                               )
                     )

    common.log(func_id+" Returning Generator...")
     
    return generator                   
                                      
#-----------------------------------

def _generate_RS1GG(step, evt_type, energy, evtnumber):
    """
    Here the settings for the RS1 graviton into gamma gamma.
    """
      
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")         
    
    # Build the process source
    generator = cms.EDFilter('Pythia6GeneratorFilter', 
                      pythiaPylistVerbosity=cms.untracked.int32(0),
                      pythiaHepMCVerbosity=cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0), 
                      filterEfficiency = cms.untracked.double(1),    
                      PythiaParameters = cms.PSet\
                               (parameterSets = cms.vstring('PythiaUESettings','processParameters'),
                                PythiaUESettings=user_pythia_ue_settings(),
                                processParameters=\
                                    cms.vstring('MSEL=0   ', 
                                                'MSUB(391)   =1   ', 
                                                'MSUB(392)   =1   ', 
                                                'PMAS(347,1) = %s ' %energy,# ! minv ', 
                                                'PARP(50)    = 0.54 ',# ! 0.54 == c=0.1', 
                                                'MDME(4158,1)=0   ',
                                                'MDME(4159,1)=0   ',
                                                'MDME(4160,1)=0   ',
                                                'MDME(4161,1)=0   ',
                                                'MDME(4162,1)=0   ',
                                                'MDME(4163,1)=0   ',
                                                'MDME(4164,1)=0   ',
                                                'MDME(4165,1)=0   ',
                                                'MDME(4166,1)=0   ',
                                                'MDME(4167,1)=0   ',
                                                'MDME(4168,1)=0   ',
                                                'MDME(4169,1)=0   ',
                                                'MDME(4170,1)=0   ',
                                                'MDME(4170,1)=0   ',
                                                'MDME(4171,1)=0   ',
                                                'MDME(4172,1)=0   ',
                                                'MDME(4173,1)=0   ',
                                                'MDME(4174,1)=0   ',
                                                'MDME(4175,1)=1   ',#! gamma gamma ', 
                                                'MDME(4176,1)=0   ', 
                                                'MDME(4177,1)=0   ', 
                                                'MDME(4178,1)=0   ', 
                                                'CKIN(3)=20.      ',#! minimum pt hat for hard interactions', 
                                                'CKIN(4)=-1.      '#! maximum pt hat for hard interactions'
                                               )
                               )
                     )

    common.log(func_id+" Returning Generator...")
     
    return generator                     
#-----------------------------------

def _generate_HpT(step, evt_type, energy, evtnumber):
    """
    Here the settings for the RS1 graviton into gamma gamma.
    """
      
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ")         
    
    # Build the process source
    generator = cms.EDFilter("Pythia6GeneratorFilter",
                      pythiaPylistVerbosity = cms.untracked.int32(0),
                      pythiaHepMCVerbosity = cms.untracked.bool(False),
                      maxEventsToPrint = cms.untracked.int32(0),
                      filterEfficiency = cms.untracked.double(1.0),
                      PythiaParameters = cms.PSet(\
                       parameterSets = cms.vstring('PythiaUESettings', 'processParameters', 'pythiaMSSMmhmax'),
                       PythiaUESettings=user_pythia_ue_settings(),
                       processParameters=cms.vstring\
                               ('MSEL = 0       ',#         ! user control', 
                                'MSUB(401) = 1  ',#         ! gg->tbH+ Registered by Alexandre.Nikitenko@cern.ch', 
                                'MSUB(402) = 1  ',#         ! qq->tbH+ Registered by Alexandre.Nikitenko@cern.ch', 
                                'IMSS(1)= 1     ',#         ! MSSM ', 'RMSS(5) = 30.           ! TANBETA', 
                                'RMSS(19) = 200.',#         ! (D=850.) m_A', 
                                'MDME(503,1)=0  ',#         !Higgs(H+) decay into dbar            u', 
                                'MDME(504,1)=0  ',#         !Higgs(H+) decay into sbar            c', 
                                'MDME(505,1)=0  ',#         !Higgs(H+) decay into bbar            t', 
                                'MDME(506,1)=0  ',#         !Higgs(H+) decay into b bar           t', 
                                'MDME(507,1)=0  ',#         !Higgs(H+) decay into e+              nu_e', 
                                'MDME(508,1)=0  ',#         !Higgs(H+) decay into mu+             nu_mu', 
                                'MDME(509,1)=1  ',#        !Higgs(H+) decay into tau+            nu_tau', 
                                'MDME(510,1)=0  ',#         !Higgs(H+) decay into tau prime+           nu_tau', 
                                'MDME(511,1)=0  ',#         !Higgs(H+) decay into W+              h0', 
                                'MDME(512,1)=0  ',#         !Higgs(H+) decay into ~chi_10         ~chi_1+', 
                                'MDME(513,1)=0  ',#         !Higgs(H+) decay into ~chi_10         ~chi_2+', 
                                'MDME(514,1)=0  ',#         !Higgs(H+) decay into ~chi_20         ~chi_1+', 
                                'MDME(515,1)=0  ',#         !Higgs(H+) decay into ~chi_20         ~chi_2+', 
                                'MDME(516,1)=0  ',#         !Higgs(H+) decay into ~chi_30         ~chi_1+', 
                                'MDME(517,1)=0  ',#         !Higgs(H+) decay into ~chi_30         ~chi_2+', 
                                'MDME(518,1)=0  ',#         !Higgs(H+) decay into ~chi_40         ~chi_1+', 
                                'MDME(519,1)=0  ',#         !Higgs(H+) decay into ~chi_40         ~chi_2+', 
                                'MDME(520,1)=0  ',#         !Higgs(H+) decay into ~t_1            ~b_1bar', 
                                'MDME(521,1)=0  ',#         !Higgs(H+) decay into ~t_2            ~b_1bar', 
                                'MDME(522,1)=0  ',#         !Higgs(H+) decay into ~t_1            ~b_2bar', 
                                'MDME(523,1)=0  ',#         !Higgs(H+) decay into ~t_2            ~b_2bar', 
                                'MDME(524,1)=0  ',#        !Higgs(H+) decay into ~d_Lbar         ~u_L', 
                                'MDME(525,1)=0  ',#        !Higgs(H+) decay into ~s_Lbar         ~c_L', 
                                'MDME(526,1)=0  ',#         !Higgs(H+) decay into ~e_L+           ~nu_eL', 
                                'MDME(527,1)=0  ',#         !Higgs(H+) decay into ~mu_L+          ~nu_muL', 
                                'MDME(528,1)=0  ',#         !Higgs(H+) decay into ~tau_1+         ~nu_tauL', 
                                'MDME(529,1)=0  '#        !Higgs(H+) decay into ~tau_2+         ~nu_tauL'),
                               ),          
                        pythiaMSSMmhmax = cms.vstring\
                              ('RMSS(2)= 200.    ',#       ! SU(2) gaugino mass ', 
                               'RMSS(3)= 800.    ',#       ! SU(3) (gluino) mass ', 
                               'RMSS(4)= 200.    ',#      ! higgsino mass parameter', 
                               'RMSS(6)= 1000.   ',#       ! left slepton mass', 
                               'RMSS(7)= 1000.   ',#       ! right slepton mass', 
                               'RMSS(8)= 1000.   ',#       ! right slepton mass', 
                               'RMSS(9)= 1000.   ',#       ! right squark mass', 
                               'RMSS(10)= 1000.  ',#       ! left sq mass for 3th gen/heaviest stop mass', 
                               'RMSS(11)= 1000.  ',#       ! right sbottom mass/lightest sbotoom mass', 
                               'RMSS(12)= 1000.  ',#       ! right stop mass/lightest stop mass', 
                               'RMSS(13)= 1000.  ',#       ! left stau mass', 
                               'RMSS(14)= 1000.  ',#       ! right stau mass', 
                               'RMSS(15)= 2449.  ',#       ! Ab', 
                               'RMSS(16)= 2449.  ',#       ! At', 
                               'RMSS(17)= 2449.  '#       ! Atau'
                              )
                       )
                     )


    common.log(func_id+" Returning Generator...")
     
    return generator     
    
#---------------------------------    

def energy_split(energy):
    """
    Extract from a string of the form "lowenergy*highenergy" two 
    bounds. It checks on its consistency. If the format is unknown 
    the program stops.
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log( func_id+" Entering... ") 
    
    separator_list = ["-", #fault tolerance is good
                      "_",
                      "*",
                      "/",
                      ";",
                      ","]
    for separator in separator_list:
        if energy.count(separator)==1:
            common.log( func_id+" Found separator in energy string...") 
            low,high = energy.split(separator)
            if float(high) > float(low):
                common.log(func_id+" Returning Energy...")
                return (low,high)
    
    raise "Energy Format: ","Unrecognised energy format."

#-----------------------------------

def user_pythia_ue_settings():
    """
    The function simply returns a cms.vstring which is a summary of the 
    Pythia settings for the event generation
    """
    
    
    
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    common.log(func_id+" Returning PythiaUE settings...")
    
    return common.include_files('Configuration/Generator/data/PythiaUESettings.cfi')[0].pythiaUESettings
            
