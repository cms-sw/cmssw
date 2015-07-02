// system headers
#include <mutex>

// Framework Headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Fast Sim headers
#include "FastSimulation/MaterialEffects/interface/NuclearInteractionFTFSimulator.h"
#include "FastSimulation/MaterialEffects/interface/CMSDummyDeexcitation.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"

// Geant4 headers
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4TheoFSGenerator.hh"
#include "G4FTFModel.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4LundStringFragmentation.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4CascadeInterface.hh"
#include "G4DiffuseElastic.hh"

#include "G4Proton.hh"
#include "G4Neutron.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4AntiProton.hh"
#include "G4AntiNeutron.hh"
#include "G4KaonPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonZeroLong.hh"
#include "G4KaonZeroShort.hh"
#include "G4KaonZero.hh"
#include "G4AntiKaonZero.hh"
#include "G4GenericIon.hh"

#include "G4Lambda.hh"
#include "G4OmegaMinus.hh"
#include "G4SigmaMinus.hh"
#include "G4SigmaPlus.hh"
#include "G4SigmaZero.hh"
#include "G4XiMinus.hh"
#include "G4XiZero.hh"
#include "G4AntiLambda.hh"
#include "G4AntiOmegaMinus.hh"
#include "G4AntiSigmaMinus.hh"
#include "G4AntiSigmaPlus.hh"
#include "G4AntiSigmaZero.hh"
#include "G4AntiXiMinus.hh"
#include "G4AntiXiZero.hh"
#include "G4AntiAlpha.hh"
#include "G4AntiDeuteron.hh"
#include "G4AntiTriton.hh"
#include "G4AntiHe3.hh"

#include "G4Material.hh"
#include "G4DecayPhysics.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4ProcessManager.hh"
#include "G4PhysicsLogVector.hh"
#include "G4SystemOfUnits.hh"

static std::once_flag initializeOnce;
[[cms::thread_guard("initializeOnce")]] const G4ParticleDefinition* NuclearInteractionFTFSimulator::theG4Hadron[] = {0};
[[cms::thread_guard("initializeOnce")]] int NuclearInteractionFTFSimulator::theId[] = {0};

const double fact = 1.0/CLHEP::GeV;

// inelastic interaction length corrections per particle and energy 
const double corrfactors_inel[numHadrons][npoints] = {
  {1.0872, 1.1026, 1.111, 1.111, 1.0105, 0.97622, 0.9511, 0.9526, 0.97591, 0.99277, 1.0099, 1.015, 1.0217, 1.0305, 1.0391, 1.0438, 1.0397, 1.0328, 1.0232, 1.0123, 1.0},
  {1.0416, 1.1044, 1.1467, 1.1273, 1.026, 0.99085, 0.96572, 0.96724, 0.99091, 1.008, 1.0247, 1.0306, 1.0378, 1.0427, 1.0448, 1.0438, 1.0397, 1.0328, 1.0232, 1.0123, 1.0},
  {0.5308, 0.53589, 0.67059, 0.80253, 0.82341, 0.79083, 0.85967, 0.90248, 0.93792, 0.9673, 1.0034, 1.022, 1.0418, 1.0596, 1.0749, 1.079, 1.0704, 1.0576, 1.0408, 1.0214, 1.0},
  {0.49107, 0.50571, 0.64149, 0.77209, 0.80472, 0.78166, 0.83509, 0.8971, 0.93234, 0.96154, 0.99744, 1.0159, 1.0355, 1.0533, 1.0685, 1.0732, 1.0675, 1.0485, 1.0355, 1.0191, 1.0},
  {1.9746, 1.7887, 1.5645, 1.2817, 1.0187, 0.95216, 0.9998, 1.035, 1.0498, 1.0535, 1.0524, 1.0495, 1.0461, 1.0424, 1.0383, 1.0338, 1.0287, 1.0228, 1.0161, 1.0085, 1.0},
  {0.46028, 0.59514, 0.70355, 0.70698, 0.62461, 0.65103, 0.71945, 0.77753, 0.83582, 0.88422, 0.92117, 0.94889, 0.96963, 0.98497, 0.99596, 1.0033, 1.0075, 1.0091, 1.0081, 1.005, 1.0},
  {0.75016, 0.89607, 0.97185, 0.91083, 0.77425, 0.77412, 0.8374, 0.88848, 0.93104, 0.96174, 0.98262, 0.99684, 1.0065, 1.0129, 1.0168, 1.0184, 1.018, 1.0159, 1.0121, 1.0068, 1.0},
  {0.75016, 0.89607, 0.97185, 0.91083, 0.77425, 0.77412, 0.8374, 0.88848, 0.93104, 0.96174, 0.98262, 0.99684, 1.0065, 1.0129, 1.0168, 1.0184, 1.018, 1.0159, 1.0121, 1.0068, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {1.1006, 1.1332, 1.121, 1.1008, 1.086, 1.077, 1.0717, 1.0679, 1.0643, 1.0608, 1.057, 1.053, 1.0487, 1.0441, 1.0392, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1318, 1.1255, 1.1062, 1.0904, 1.0802, 1.0742, 1.0701, 1.0668, 1.0636, 1.0602, 1.0566, 1.0527, 1.0485, 1.044, 1.0391, 1.0337, 1.028, 1.0217, 1.015, 1.0078, 1.0},
  {1.1094, 1.1332, 1.1184, 1.0988, 1.0848, 1.0765, 1.0714, 1.0677, 1.0642, 1.0607, 1.0569, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1087, 1.1332, 1.1187, 1.099, 1.0849, 1.0765, 1.0715, 1.0677, 1.0642, 1.0607, 1.057, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {1.1192, 1.132, 1.1147, 1.0961, 1.0834, 1.0758, 1.0711, 1.0674, 1.064, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1188, 1.1321, 1.1149, 1.0963, 1.0834, 1.0758, 1.0711, 1.0675, 1.0641, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0.50776, 0.5463, 0.5833, 0.61873, 0.65355, 0.68954, 0.72837, 0.7701, 0.81267, 0.85332, 0.89037, 0.92329, 0.95177, 0.97539, 0.99373, 1.0066, 1.014, 1.0164, 1.0144, 1.0087, 1.0},
  {0.50787, 0.5464, 0.58338, 0.6188, 0.65361, 0.6896, 0.72841, 0.77013, 0.8127, 0.85333, 0.89038, 0.92329, 0.95178, 0.9754, 0.99373, 1.0066, 1.014, 1.0164, 1.0144, 1.0087, 1.0},
  {1.1006, 1.1332, 1.121, 1.1008, 1.086, 1.077, 1.0717, 1.0679, 1.0643, 1.0608, 1.057, 1.053, 1.0487, 1.0441, 1.0392, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1318, 1.1255, 1.1062, 1.0904, 1.0802, 1.0742, 1.0701, 1.0668, 1.0636, 1.0602, 1.0566, 1.0527, 1.0485, 1.044, 1.0391, 1.0337, 1.028, 1.0217, 1.015, 1.0078, 1.0},
  {1.1094, 1.1332, 1.1184, 1.0988, 1.0848, 1.0765, 1.0714, 1.0677, 1.0642, 1.0607, 1.0569, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1087, 1.1332, 1.1187, 1.099, 1.0849, 1.0765, 1.0715, 1.0677, 1.0642, 1.0607, 1.057, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {1.1192, 1.132, 1.1147, 1.0961, 1.0834, 1.0758, 1.0711, 1.0674, 1.064, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1188, 1.1321, 1.1149, 1.0963, 1.0834, 1.0758, 1.0711, 1.0675, 1.0641, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0.47677, 0.51941, 0.56129, 0.60176, 0.64014, 0.67589, 0.70891, 0.73991, 0.77025, 0.80104, 0.83222, 0.86236, 0.8901, 0.91518, 0.9377, 0.95733, 0.97351, 0.98584, 0.9942, 0.99879, 1.0},
  {0.49361, 0.53221, 0.56976, 0.60563, 0.63954, 0.67193, 0.70411, 0.73777, 0.77378, 0.81114, 0.84754, 0.88109, 0.91113, 0.93745, 0.95974, 0.97762, 0.99081, 0.99929, 1.0033, 1.0034, 1.0},
  {0.4873, 0.52744, 0.56669, 0.60443, 0.64007, 0.67337, 0.70482, 0.73572, 0.76755, 0.80086, 0.83456, 0.86665, 0.8959, 0.92208, 0.94503, 0.96437, 0.97967, 0.99072, 0.99756, 1.0005, 1.0},
  {0.48729, 0.52742, 0.56668, 0.60442, 0.64006, 0.67336, 0.70482, 0.73571, 0.76754, 0.80086, 0.83455, 0.86665, 0.8959, 0.92208, 0.94503, 0.96437, 0.97967, 0.99072, 0.99756, 1.0005, 1.0},
};

// elastic interaction length corrections per particle and energy 
const double corrfactors_el[numHadrons][npoints] = {
  {0.58834, 1.1238, 1.7896, 1.4409, 0.93175, 0.80587, 0.80937, 0.83954, 0.87453, 0.91082, 0.94713, 0.98195, 1.0134, 1.0397, 1.0593, 1.071, 1.0739, 1.0678, 1.053, 1.03, 1.0},
  {0.40938, 0.92337, 1.3365, 1.1607, 1.008, 0.82206, 0.81163, 0.79489, 0.82919, 0.91812, 0.96688, 1.0225, 1.0734, 1.0833, 1.0874, 1.0854, 1.0773, 1.0637, 1.0448, 1.0235, 1.0},
  {0.43699, 0.42165, 0.46594, 0.64917, 0.85314, 0.80782, 0.83204, 0.91162, 1.0155, 1.0665, 1.0967, 1.1125, 1.1275, 1.1376, 1.1464, 1.1477, 1.1312, 1.1067, 1.0751, 1.039, 1.0},
  {0.3888, 0.39527, 0.43921, 0.62834, 0.8164, 0.79866, 0.82272, 0.90163, 1.0045, 1.055, 1.0849, 1.1005, 1.1153, 1.1253, 1.134, 1.1365, 1.1255, 1.0895, 1.0652, 1.0348, 1.0},
  {0.32004, 0.31119, 0.30453, 0.30004, 0.31954, 0.40148, 0.5481, 0.74485, 0.99317, 1.1642, 1.2117, 1.2351, 1.2649, 1.3054, 1.375, 1.4992, 1.4098, 1.3191, 1.2232, 1.118, 1.0},
  {0.10553, 0.14623, 0.20655, 0.26279, 0.19996, 0.40125, 0.5139, 0.71271, 0.89269, 1.0108, 1.1673, 1.3052, 1.4149, 1.429, 1.4521, 1.4886, 1.4006, 1.3116, 1.2177, 1.1151, 1.0},
  {0.106, 0.14692, 0.20755, 0.26257, 0.20089, 0.40236, 0.51452, 0.71316, 0.89295, 1.0109, 1.1673, 1.3053, 1.4149, 1.429, 1.4521, 1.4886, 1.4006, 1.3116, 1.2177, 1.1151, 1.0},
  {0.31991, 0.31111, 0.30445, 0.30004, 0.31995, 0.40221, 0.54884, 0.74534, 0.99364, 1.1644, 1.2117, 1.2351, 1.265, 1.3054, 1.375, 1.4992, 1.4098, 1.3191, 1.2232, 1.118, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {0.37579, 0.39922, 0.37445, 0.32631, 0.39002, 0.42161, 0.54251, 0.69127, 0.90332, 1.0664, 1.1346, 1.1481, 1.1692, 1.2036, 1.2625, 1.3633, 1.2913, 1.2215, 1.1516, 1.0788, 1.0},
  {0.31756, 0.33409, 0.25339, 0.35525, 0.52989, 0.63382, 0.7453, 0.93505, 1.1464, 1.2942, 1.3161, 1.328, 1.3393, 1.3525, 1.374, 1.4051, 1.3282, 1.2523, 1.1745, 1.0916, 1.0},
  {0.38204, 0.39694, 0.36502, 0.33367, 0.39229, 0.43119, 0.54898, 0.70169, 0.91004, 1.0696, 1.1348, 1.1483, 1.1694, 1.2038, 1.2627, 1.3632, 1.2913, 1.2215, 1.1516, 1.0788, 1.0},
  {0.38143, 0.39716, 0.36609, 0.33294, 0.39207, 0.43021, 0.54834, 0.70066, 0.90945, 1.0693, 1.1348, 1.1482, 1.1694, 1.2038, 1.2627, 1.3632, 1.2913, 1.2215, 1.1516, 1.0788, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {0.29564, 0.32645, 0.29986, 0.30611, 0.48808, 0.59902, 0.71207, 0.8832, 1.1164, 1.2817, 1.3154, 1.3273, 1.3389, 1.3521, 1.3736, 1.4056, 1.3285, 1.2524, 1.1746, 1.0916, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {0.3265, 0.3591, 0.39232, 0.42635, 0.46259, 0.50365, 0.55244, 0.61014, 0.67446, 0.74026, 0.80252, 0.85858, 0.90765, 0.94928, 0.9827, 1.0071, 1.0221, 1.0279, 1.0253, 1.0155, 1.0},
  {0.13808, 0.15585, 0.17798, 0.2045, 0.22596, 0.25427, 0.33214, 0.44821, 0.5856, 0.74959, 0.89334, 1.0081, 1.0964, 1.1248, 1.173, 1.2548, 1.1952, 1.1406, 1.0903, 1.0437, 1.0},
  {0.20585, 0.23253, 0.26371, 0.28117, 0.30433, 0.35417, 0.44902, 0.58211, 0.73486, 0.90579, 1.0395, 1.1488, 1.2211, 1.2341, 1.2553, 1.2877, 1.2245, 1.1654, 1.1093, 1.0547, 1.0},
  {0.2852, 0.32363, 0.31419, 0.35164, 0.45463, 0.54331, 0.66908, 0.81735, 0.98253, 1.1557, 1.2557, 1.3702, 1.4186, 1.401, 1.374, 1.3325, 1.2644, 1.1991, 1.1348, 1.0694, 1.0},
  {0.20928, 0.23671, 0.2664, 0.28392, 0.30584, 0.35929, 0.45725, 0.5893, 0.74047, 0.9101, 1.0407, 1.1503, 1.2212, 1.2342, 1.2554, 1.2876, 1.2245, 1.1654, 1.1093, 1.0547, 1.0},
  {0.11897, 0.13611, 0.15796, 0.1797, 0.21335, 0.26367, 0.34705, 0.46115, 0.6016, 0.7759, 0.91619, 1.0523, 1.1484, 1.1714, 1.2098, 1.2721, 1.2106, 1.1537, 1.1004, 1.0496, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {0.26663, 0.30469, 0.32886, 0.33487, 0.41692, 0.51616, 0.63323, 0.78162, 0.95551, 1.1372, 1.2502, 1.3634, 1.4189, 1.4013, 1.3743, 1.3329, 1.2646, 1.1992, 1.1349, 1.0694, 1.0},
  {0.16553, 0.19066, 0.21468, 0.23609, 0.30416, 0.38821, 0.49644, 0.63386, 0.80299, 0.99907, 1.1304, 1.2724, 1.3535, 1.3475, 1.3381, 1.3219, 1.2549, 1.191, 1.1287, 1.0659, 1.0},
  {0.37736, 0.41414, 0.45135, 0.48843, 0.52473, 0.55973, 0.59348, 0.62696, 0.66202, 0.70042, 0.74241, 0.786, 0.82819, 0.86688, 0.90128, 0.93107, 0.95589, 0.97532, 0.98908, 0.99719, 1.0},
  {0.34354, 0.37692, 0.4109, 0.44492, 0.47873, 0.51296, 0.54937, 0.59047, 0.63799, 0.69117, 0.74652, 0.7998, 0.84832, 0.89111, 0.92783, 0.95798, 0.98095, 0.99635, 1.0043, 1.0052, 1.0},
  {0.36364, 0.39792, 0.43277, 0.4676, 0.50186, 0.53538, 0.56884, 0.604, 0.64308, 0.68729, 0.73544, 0.7842, 0.83019, 0.87156, 0.90777, 0.93854, 0.96346, 0.98209, 0.99421, 1, 1.0},
  {0.36362, 0.39791, 0.43276, 0.46759, 0.50185, 0.53537, 0.56883, 0.604, 0.64307, 0.68728, 0.73544, 0.7842, 0.83019, 0.87156, 0.90777, 0.93854, 0.96346, 0.98209, 0.99421, 1, 1.0},
};

// inelastic interaction length in Silicon at 1 TeV per particle
const double nuclInelLength[numHadrons] = {
4.5606, 4.4916, 5.7511, 5.7856, 6.797, 6.8373, 6.8171, 6.8171, 0, 0, 4.6926, 4.6926, 4.6926, 4.6926, 0, 4.6926, 4.6926, 4.3171, 4.3171, 4.6926, 4.6926, 4.6926, 4.6926, 0, 4.6926, 4.6926, 2.509, 2.9048, 2.5479, 2.5479
};

// elastic interaction length in Silicon at 1 TeV per particle
const double nuclElLength[numHadrons] = {
9.248, 9.451, 11.545, 11.671, 32.081, 34.373, 34.373, 32.081, 0, 0, 15.739, 20.348, 15.739, 15.739, 0, 20.349, 0, 9.7514, 12.864, 15.836, 20.516, 15.836, 15.744, 0, 20.517, 20.44, 4.129, 6.0904, 4.5204, 4.5204
};


NuclearInteractionFTFSimulator::NuclearInteractionFTFSimulator(  
  unsigned int distAlgo, double distCut, double elimit, double eth) :
  curr4Mom(0.,0.,0.,0.),
  vectProj(0.,0.,1.),
  theBoost(0.,0.,0.),
  theBertiniLimit(elimit),
  theEnergyLimit(eth),
  theDistCut(distCut),
  distMin(1E99),
  theDistAlgo(distAlgo)
{
  // FTF model
  theHadronicModel = new G4TheoFSGenerator("FTF");
  theStringModel = new G4FTFModel();
  G4GeneratorPrecompoundInterface* cascade 
    = new G4GeneratorPrecompoundInterface(new CMSDummyDeexcitation());
  theLund = new G4LundStringFragmentation();
  theStringDecay = new G4ExcitedStringDecay(theLund);
  theStringModel->SetFragmentationModel(theStringDecay);

  theHadronicModel->SetTransport(cascade);
  theHadronicModel->SetHighEnergyGenerator(theStringModel);
  theHadronicModel->SetMinEnergy(theEnergyLimit);

  // Bertini Cascade 
  theBertiniCascade = new G4CascadeInterface();

  theDiffuseElastic = new G4DiffuseElastic();

  // Geant4 particles and cross sections
  std::call_once(initializeOnce, [this] () {
    theG4Hadron[0] = G4Proton::Proton();
    theG4Hadron[1] = G4Neutron::Neutron();
    theG4Hadron[2] = G4PionPlus::PionPlus();
    theG4Hadron[3] = G4PionMinus::PionMinus();
    theG4Hadron[4] = G4KaonPlus::KaonPlus();
    theG4Hadron[5] = G4KaonMinus::KaonMinus();
    theG4Hadron[6] = G4KaonZeroLong::KaonZeroLong();
    theG4Hadron[7] = G4KaonZeroShort::KaonZeroShort();
    theG4Hadron[8] = G4KaonZero::KaonZero();
    theG4Hadron[9] = G4AntiKaonZero::AntiKaonZero();
    theG4Hadron[10]= G4Lambda::Lambda();
    theG4Hadron[11]= G4OmegaMinus::OmegaMinus();
    theG4Hadron[12]= G4SigmaMinus::SigmaMinus();
    theG4Hadron[13]= G4SigmaPlus::SigmaPlus();
    theG4Hadron[14]= G4SigmaZero::SigmaZero();
    theG4Hadron[15]= G4XiMinus::XiMinus();
    theG4Hadron[16]= G4XiZero::XiZero();
    theG4Hadron[17]= G4AntiProton::AntiProton();
    theG4Hadron[18]= G4AntiNeutron::AntiNeutron();
    theG4Hadron[19]= G4AntiLambda::AntiLambda();
    theG4Hadron[20]= G4AntiOmegaMinus::AntiOmegaMinus();
    theG4Hadron[21]= G4AntiSigmaMinus::AntiSigmaMinus();
    theG4Hadron[22]= G4AntiSigmaPlus::AntiSigmaPlus();
    theG4Hadron[23]= G4AntiSigmaZero::AntiSigmaZero();
    theG4Hadron[24]= G4AntiXiMinus::AntiXiMinus();
    theG4Hadron[25]= G4AntiXiZero::AntiXiZero();
    theG4Hadron[26]= G4AntiAlpha::AntiAlpha();
    theG4Hadron[27]= G4AntiDeuteron::AntiDeuteron();
    theG4Hadron[28]= G4AntiTriton::AntiTriton();
    theG4Hadron[29]= G4AntiHe3::AntiHe3();

    // other Geant4 particles
    G4ParticleDefinition* ion = G4GenericIon::GenericIon();
    ion->SetProcessManager(new G4ProcessManager(ion));
    G4DecayPhysics decays;
    decays.ConstructParticle();  
    G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
    partTable->SetVerboseLevel(0);
    partTable->SetReadiness();

    for(int i=0; i<numHadrons; ++i) {
      theId[i] = theG4Hadron[i]->GetPDGEncoding();
    }
  }); 

  // local objects
  vect = new G4PhysicsLogVector(npoints-1,100*MeV,TeV);
  intLengthElastic = intLengthInelastic = 0.0;
  currIdx = 0;
  index = 0;
  currTrack = 0;
  currParticle = theG4Hadron[0];

  // fill projectile particle definitions
  dummyStep = new G4Step();
  dummyStep->SetPreStepPoint(new G4StepPoint());

  // target is always Silicon
  targetNucleus.SetParameters(28, 14);
}

NuclearInteractionFTFSimulator::~NuclearInteractionFTFSimulator() {

  delete theStringDecay;
  delete theStringModel;
  delete theLund;
  delete vect;
}

void NuclearInteractionFTFSimulator::compute(ParticlePropagator& Particle, 
					     RandomEngineAndDistribution const* random)
{
  //std::cout << "#### Primary " << Particle.pid() << " E(GeV)= " 
  //	    << Particle.momentum().e() << std::endl;

  int thePid = Particle.pid(); 
  if(thePid != theId[currIdx]) {
    currParticle = 0;
    currIdx = 0;
    for(; currIdx<numHadrons; ++currIdx) {
      if(theId[currIdx] == thePid) {
	currParticle = theG4Hadron[currIdx];
	// neutral kaons
	if(7 == currIdx || 8 == currIdx) {
	  currParticle = theG4Hadron[9];
	  if(random->flatShoot() > 0.5) { currParticle = theG4Hadron[10]; }
	}
	break;
      }
    }
  }
  if(!currParticle) { return; }

  // fill projectile for Geant4
  double mass = currParticle->GetPDGMass();
  double ekin = CLHEP::GeV*Particle.momentum().e() - mass;

  // check interaction length
  intLengthElastic   = nuclElLength[currIdx];
  intLengthInelastic = nuclInelLength[currIdx];
  if(0.0 == intLengthInelastic) { return; }

  // apply corrections
  if(ekin <= vect->Energy(0)) {
    intLengthElastic   *= corrfactors_el[currIdx][0];
    intLengthInelastic *= corrfactors_inel[currIdx][0];
  } else if(ekin < vect->Energy(npoints-1)) {
    index = vect->FindBin(ekin, index);
    double e1 = vect->Energy(index);
    double e2 = vect->Energy(index+1);
    intLengthElastic   *= ((corrfactors_el[currIdx][index]*(e2 - ekin) + 
			    corrfactors_el[currIdx][index+1]*(ekin - e1))/(e2 - e1));
    intLengthInelastic *= ((corrfactors_inel[currIdx][index]*(e2 - ekin) + 
			    corrfactors_inel[currIdx][index+1]*(ekin - e1))/(e2 - e1));
  }
  /*
  std::cout << " Primary " <<  currParticle->GetParticleName() 
  	    << "  E(GeV)= " << e*fact << std::endl;
  */

  double currInteractionLength = -G4Log(random->flatShoot())*intLengthElastic*intLengthInelastic
    /(intLengthElastic + intLengthInelastic); 
  /*
  std::cout << "*NuclearInteractionFTFSimulator::compute: R(X0)= " << radLengths
	    << " Rnuc(X0)= " << theNuclIntLength[currIdx] << "  IntLength(X0)= " 
            << currInteractionLength << std::endl;
  */
  // Check position of nuclear interaction
  if (currInteractionLength > radLengths) { return; }

  // fill projectile for Geant4
  double px = Particle.momentum().px();
  double py = Particle.momentum().py();
  double pz = Particle.momentum().pz();
  double ptot = sqrt(px*px + py*py + pz*pz);
  double norm = 1./ptot;
  G4ThreeVector dir(px*norm, py*norm, pz*norm);
  /*
  std::cout << " Primary " <<  currParticle->GetParticleName() 
	    << "  E(GeV)= " << e*fact << "  P(GeV/c)= (" 
	    << px << " " << py << " " << pz << ")" << std::endl;
  */

  G4DynamicParticle* dynParticle = new G4DynamicParticle(theG4Hadron[currIdx],dir,ekin);
  currTrack = new G4Track(dynParticle, 0.0, vectProj);
  currTrack->SetStep(dummyStep);

  theProjectile.Initialise(*currTrack); 
  delete currTrack;

  G4HadFinalState* result;

  // elastic interaction
  if(random->flatShoot()*(intLengthElastic + intLengthInelastic) > intLengthElastic) {

    result = theDiffuseElastic->ApplyYourself(theProjectile, targetNucleus);
    G4ThreeVector dirnew = result->GetMomentumChange().unit();
    double cost = (dir*dirnew);
    double sint = std::sqrt((1. - cost)*(1. + cost));

    curr4Mom.set(ptot*dirnew.x(),ptot*dirnew.y(),ptot*dirnew.z(),Particle.momentum().e());

    // Always create a daughter if the kink is large engough 
    if (sint > theDistCut) { 
      saveDaughter(Particle, curr4Mom, thePid); 
    } else {
      Particle.SetXYZT(curr4Mom.px(), curr4Mom.py(), curr4Mom.pz(), curr4Mom.e());
    }

    // inelastic interaction
  } else {

    // Bertini cascade for low-energy hadrons (except light anti-nuclei)
    // FTFP is applied above energy limit and for all anti-hyperons and anti-ions 
    if(ekin <= theBertiniLimit && currIdx < 17) { 
      result = theBertiniCascade->ApplyYourself(theProjectile, targetNucleus);
    } else {
      result = theHadronicModel->ApplyYourself(theProjectile, targetNucleus);
    }
    if(result) {

      int nsec = result->GetNumberOfSecondaries();
      if(0 < nsec) {

	result->SetTrafoToLab(theProjectile.GetTrafoToLab());
	_theUpdatedState.clear();

	//std::cout << "   " << nsec << " secondaries" << std::endl;
	// Generate angle
	double phi = random->flatShoot()*CLHEP::twopi;
	theClosestChargedDaughterId = -1;
	distMin = 1e99;

	// rotate and store secondaries
	for (int j=0; j<nsec; ++j) {

	  const G4DynamicParticle* dp = result->GetSecondary(j)->GetParticle();
	  int thePid = dp->GetParticleDefinition()->GetPDGEncoding();

	  // rotate around primary direction
	  curr4Mom = dp->Get4Momentum();
	  curr4Mom.rotate(phi, vectProj);
	  curr4Mom *= result->GetTrafoToLab();
	  /*
	    std::cout << j << ". " << dp->GetParticleDefinition()->GetParticleName() 
	    << "  " << thePid
	    << "  " << curr4Mom*fact << std::endl;
	  */
	  // prompt 2-gamma decay for pi0, eta, eta'p
	  if(111 == thePid || 221 == thePid || 331 == thePid) {
	    theBoost = curr4Mom.boostVector();
	    double e = 0.5*dp->GetParticleDefinition()->GetPDGMass();
	    double fi  = random->flatShoot()*CLHEP::twopi; 
	    double cth = 2*random->flatShoot() - 1.0;
	    double sth = sqrt((1.0 - cth)*(1.0 + cth)); 
	    G4LorentzVector lv(e*sth*cos(fi),e*sth*sin(fi),e*cth,e);
	    lv.boost(theBoost);
	    saveDaughter(Particle, lv, 22); 
	    curr4Mom -= lv;
	    if(curr4Mom.e() > theEnergyLimit) { 
	      saveDaughter(Particle, curr4Mom, 22); 
	    } 
	  } else {
	    if(curr4Mom.e() > theEnergyLimit + dp->GetParticleDefinition()->GetPDGMass()) { 
	      saveDaughter(Particle, curr4Mom, thePid); 
	    }
	  }
	}
	result->Clear();
      }
    }
  }
}

void NuclearInteractionFTFSimulator::saveDaughter(ParticlePropagator& Particle, 
						  const G4LorentzVector& lv, int pdgid)
{
  unsigned int idx = _theUpdatedState.size();   
  _theUpdatedState.push_back(Particle);
  _theUpdatedState[idx].SetXYZT(lv.px()*fact,lv.py()*fact,lv.pz()*fact,lv.e()*fact);
  _theUpdatedState[idx].setID(pdgid);

  // Store the closest daughter index (for later tracking purposes, so charged particles only) 
  double distance = distanceToPrimary(Particle,_theUpdatedState[idx]);
  // Find the closest daughter, if closer than a given upper limit.
  if ( distance < distMin && distance < theDistCut ) {
    distMin = distance;
    theClosestChargedDaughterId = idx;
  }
  // std::cout << _theUpdatedState[idx] << std::endl;
}

double 
NuclearInteractionFTFSimulator::distanceToPrimary(const RawParticle& Particle,
						  const RawParticle& aDaughter) const 
{
  double distance = 2E99;
  // Compute the distance only for charged primaries
  if ( fabs(Particle.charge()) > 1E-6 ) { 

    // The secondary must have the same charge
    double chargeDiff = fabs(aDaughter.charge()-Particle.charge());
    if ( fabs(chargeDiff) < 1E-6 ) {

      // Here are two distance definitions * to be tuned *
      switch ( theDistAlgo ) { 
	
      case 1:
	// sin(theta12)
	distance = (aDaughter.Vect().Unit().Cross(Particle.Vect().Unit())).R();
	break;
	
      case 2: 
	// sin(theta12) * p1/p2
	distance = (aDaughter.Vect().Cross(Particle.Vect())).R()
	  /aDaughter.Vect().Mag2();
	break;
	
      default:
	// Should not happen
	break;	
      }
    }
  } 
  return distance;
}
