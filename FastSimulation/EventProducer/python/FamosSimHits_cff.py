import FWCore.ParameterSet.Config as cms

# Here so that python translator can see the names
# Now beta function vertex smearing 
from FastSimulation.Event.EarlyCollisionVertexGenerator_cfi import *
# include "FastSimulation/Event/data/NominalCollisionVertexGenerator.cfi"
# include "FastSimulation/Event/data/NominalCollision1VertexGenerator.cfi"
# include "FastSimulation/Event/data/NominalCollision2VertexGenerator.cfi"
# include "FastSimulation/Event/data/NominalCollision3VertexGenerator.cfi"
# include "FastSimulation/Event/data/NominalCollision4VertexGenerator.cfi"
# include "FastSimulation/Event/data/GaussianVertexGenerator.cfi"
# include "FastSimulation/Event/data/FlatVertexGenerator.cfi"
# include "FastSimulation/Event/data/NoVertexGenerator.cfi"
# Kinematic cuts for the particle filter in the SimEvent
from FastSimulation.Event.ParticleFilter_cfi import *
# Material effects to be simulated in the tracker material and associated cuts
from FastSimulation.MaterialEffects.MaterialEffects_cfi import *
# (De)activate decays of unstable particles (K0S, etc...)
from FastSimulation.TrajectoryManager.ActivateDecays_cfi import *
# Conditions to save Tracker SimHits 
from FastSimulation.TrajectoryManager.TrackerSimHits_cfi import *
# FastCalorimetry
from FastSimulation.Calorimetry.Calorimetry_cff import *
famosSimHits = cms.EDProducer("FamosProducer",
    SimulateCalorimetry = cms.bool(True),
    Calorimetry = cms.PSet(
        ECAL = cms.PSet(
            CoreIntervals = cms.vdouble(100.0, 0.1),
            RTFactor = cms.double(1.0),
            RCFactor = cms.double(1.0),
            TailIntervals = cms.vdouble(1.0, 0.1, 100.0, 1.0),
            FrontLeakageProbability = cms.double(1.0),
            GridSize = cms.int32(7),
            RadiusFactor = cms.double(1.096),
            Debug = cms.untracked.bool(False),
            SpotFraction = cms.double(-1.0),
            GapLossProbability = cms.double(0.9)
        ),
        CalorimeterProperties = cms.PSet(
            PreshowerLayer2_thickness = cms.double(0.38),
            ECALEndcap_LightCollection = cms.double(0.023),
            PreshowerLayer1_thickness = cms.double(1.6),
            PreshowerLayer1_mipsPerGeV = cms.double(35.7),
            PreshowerLayer2_mipsPerGeV = cms.double(59.5),
            ECALBarrel_LightCollection = cms.double(0.03),
            HCAL_Sampling = cms.double(0.0035),
            HCAL_PiOverE = cms.double(0.2)
        ),
        UnfoldedMode = cms.untracked.bool(False),
        HCAL = cms.PSet(
            SimMethod = cms.int32(0),
            GridSize = cms.int32(7),
            SimOption = cms.int32(2)
        ),
        HSParameters = cms.PSet(
            nTRsteps = cms.int32(40),
            lossesOpt = cms.int32(0),
            depthStep = cms.double(0.5),
            balanceEH = cms.double(0.9),
            eSpotSize = cms.double(0.2),
            hcalDepthFactor = cms.double(1.1),
            transRparam = cms.double(1.0),
            nDepthSteps = cms.int32(10),
            maxTRfactor = cms.double(4.0),
            criticalHDEnergy = cms.double(3.0)
        ),
        HCALResponse = cms.PSet(
            eResponseCoefficient = cms.double(1.0),
            HadronEndcapResolution_Noise = cms.double(0.0),
            HadronForwardResolution_Stochastic = cms.double(1.82),
            ElectronForwardResolution_Constant = cms.double(0.05),
            HadronBarrelResolution_Noise = cms.double(0.0),
            HadronForwardResolution_Constant = cms.double(0.09),
            HadronBarrelResolution_Stochastic = cms.double(1.22),
            HadronEndcapResolution_Constant = cms.double(0.05),
            eResponseExponent = cms.double(1.0),
            HadronForwardResolution_Noise = cms.double(0.0),
            HadronBarrelResolution_Constant = cms.double(0.05),
            HadronEndcapResolution_Stochastic = cms.double(1.3),
            eResponseCorrection = cms.double(1.0),
            eResponseScaleHB = cms.double(3.0),
            eResponseScaleHF = cms.double(3.0),
            eResponseScaleHE = cms.double(3.0),
            ElectronForwardResolution_Stochastic = cms.double(1.38),
            eResponsePlateauHE = cms.double(0.95),
            eResponsePlateauHF = cms.double(0.95),
            eResponsePlateauHB = cms.double(0.95),
            energyBias = cms.double(0.0),
            ElectronForwardResolution_Noise = cms.double(0.0)
        )
    ),
    SimulateMuons = cms.bool(True),
    RunNumber = cms.untracked.int32(1001),
    Verbosity = cms.untracked.int32(0),
    ActivateDecays = cms.PSet(
        ActivateDecays = cms.bool(True)
    ),
    UseMagneticField = cms.bool(True),
    MaterialEffects = cms.PSet(
        Bremsstrahlung = cms.bool(True),
        distCut = cms.double(0.02),
        pionMasses = cms.untracked.vdouble(0.13957, 0.13957, 0.497648, 0.493677, 0.493677, 0.93827, 0.93827, 0.939565, 0.939565),
        K0Ls = cms.untracked.vint32(130, 310),
        NuclearInteraction = cms.bool(True),
        Kplusses = cms.untracked.vint32(321),
        antiprotons = cms.untracked.vint32(-2212, -3222),
        pionEnergies = cms.untracked.vdouble(1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 1000.0),
        distAlgo = cms.uint32(1),
        PairProduction = cms.bool(True),
        inputFile = cms.untracked.string('NuclearInteractionInputFile.txt'),
        Piminusses = cms.untracked.vint32(-211),
        antineutrons = cms.untracked.vint32(-2112, -3122, -3112, -3312, -3322, -3334),
        photonEnergy = cms.double(0.1),
        bremEnergyFraction = cms.double(0.005),
        MultipleScattering = cms.bool(True),
        pionTypes = cms.untracked.vint32(211, -211, 130, 321, -321, 2212, -2212, 2112, -2112),
        ratios = cms.untracked.vdouble(0.031390573, 0.531842852, 0.819614219, 0.951251711, 0.98638275, 1.0, 0.985087033, 0.982996773, 0.990832192, 0.992237923, 0.99484158, 0.973816742, 0.967264815, 0.971714258, 0.969122824, 0.978681792, 0.977312732, 0.984255819, 0.035326512, 0.577356403, 0.857118809, 0.965683504, 0.98965936, 1.0, 0.98959924, 0.980665408, 0.988384816, 0.981038152, 0.975002104, 0.959996152, 0.953310808, 0.954705592, 0.9576154, 0.961150456, 0.965022184, 0.960573304, 0.0, 0.370261189, 0.649793096, 0.734342408, 0.749079499, 0.753360057, 0.755790543, 0.755872164, 0.751337674, 0.746685288, 0.747519634, 0.739357554, 0.735004444, 0.803039922, 0.832749896, 0.890900187, 0.936734805, 1.0, 0.0, 0.175571717, 0.391683394, 0.528946472, 0.572818635, 0.61421028, 0.644125538, 0.67030405, 0.685144573, 0.702870161, 0.714708513, 0.730805263, 0.777711536, 0.831090576, 0.869267129, 0.915747562, 0.953370523, 1.0, 0.0, 0.36535321, 0.611663677, 0.715315908, 0.733498956, 0.738361302, 0.745253654, 0.751459671, 0.750628335, 0.746442657, 0.750850669, 0.744895986, 0.73509396, 0.791663444, 0.828609543, 0.88999304, 0.940897842, 1.0, 0.0, 0.042849136, 0.459103223, 0.666165343, 0.787930873, 0.890397011, 0.920999533, 0.937832788, 0.950920131, 0.966595049, 0.97954227, 0.988061653, 0.983260159, 0.988958431, 0.991723494, 0.995273237, 1.0, 0.999962634, 1.0, 0.849956907, 0.775625988, 0.80201823, 0.816207485, 0.785899785, 0.754998487, 0.728977244, 0.710010673, 0.670890339, 0.665627872, 0.652682888, 0.613334247, 0.647534574, 0.667910938, 0.689919693, 0.709200185, 0.724199928, 0.0, 0.059216484, 0.437844536, 0.610370629, 0.702090648, 0.78007689, 0.802143073, 0.819570432, 0.825829666, 0.84007975, 0.838435509, 0.837529986, 0.835687165, 0.885205014, 0.912450156, 0.951451221, 0.973215562, 1.0, 1.0, 0.849573257, 0.756479495, 0.787147094, 0.804572414, 0.791806302, 0.760234588, 0.741109531, 0.724118186, 0.692829761, 0.688465897, 0.671806061, 0.636461171, 0.675314029, 0.69913446, 0.724305037, 0.742556115, 0.758504713),
        fudgeFactor = cms.double(1.2),
        bremEnergy = cms.double(0.1),
        pionMinP = cms.untracked.vdouble(0.7, 0.0, 1.0, 1.0, 0.0, 1.1, 0.0, 1.1, 0.0),
        neutrons = cms.untracked.vint32(2112, 3122, 3112, 3312, 3322, 3334),
        lengthRatio = cms.vdouble(0.2257, 0.2294, 0.3042, 0.2591, 0.2854, 0.3101, 0.5216, 0.3668, 0.4898),
        protons = cms.untracked.vint32(2212, 3222, -101, -102, -103, -104),
        pionEnergy = cms.double(0.5),
        pTmin = cms.double(0.3),
        Piplusses = cms.untracked.vint32(211),
        EnergyLoss = cms.bool(True),
        Kminusses = cms.untracked.vint32(-321),
        pionNames = cms.untracked.vstring('piplus', 'piminus', 'K0L', 'Kplus', 'Kminus', 'p', 'pbar', 'n', 'nbar')
    ),
    ParticleFilter = cms.PSet(
        EProton = cms.double(5000.0),
        etaMax = cms.double(5.0),
        pTMin = cms.double(0.2),
        EMin = cms.double(0.1)
    ),
    UseTRandomEngine = cms.bool(True),
    TrackerSimHits = cms.PSet(
        pTmin = cms.untracked.double(0.3),
        firstLoop = cms.untracked.bool(True)
    ),
    SimulateTracking = cms.bool(True),
    ApplyAlignment = cms.bool(False),
    VertexGenerator = cms.PSet(
        myVertexGenerator
    )
)


