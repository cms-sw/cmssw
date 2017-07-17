/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#include "GeneratorInterface/Hydjet2Interface/interface/ParticleTable.h"

//This table is an additional table to describe the particle properties
// that are missing in the ROOT pdg table
//There are the particles and resonances from ROOT pdg-table.
//Only wellknown and consisting of u,d,s quarks states are included.
// 
//code: Baryon number, Strangeness, 2*Isospin, 2*Spin, Charge 

typedef std::pair<const int, ParticleInfo> Part_t;

const Part_t gTableInitializer[] = 
  {
    //     pdg                 B  S  2I 2Sp Ch         mass   width   name  
    Part_t(10331, ParticleInfo(0, 0, 0, 0, 0)),   //   1.715, 0.125, f01710zer 
    Part_t(3334,  ParticleInfo(1, -3, 0, 3, -1)), //   1.67245, 8.003e-16, UM1672min
    Part_t(-3334, ParticleInfo(-1, 3, 0, 3, 1)),  //   1.67245, 8.003e-16, UM1672mnb
    Part_t(3324,  ParticleInfo(1, -2, 1, 3, -1)), //   1.535, 0.0099, Xi1530min
    Part_t(-3324, ParticleInfo(-1, 2, 1, 3, 1)),  //   1.535, 0.0099, Xi1530mnb
    Part_t(3314,  ParticleInfo(1, -2, 1, 3, 0)),  //   1.5318, 0.0091, Xi1530zer
    Part_t(-3314, ParticleInfo(-1, 2, 1, 3, 0)),  //   1.5318, 0.0091, Xi1530zrb
    Part_t(335,   ParticleInfo(0, 0, 0, 4, 0)),   //   1.525, 0.076, f21525zer
    Part_t(10211, ParticleInfo(0, 0, 2, 0, 1)),   //   1.474, 0.265, a01450plu
    Part_t(-10211,ParticleInfo(0, 0, 2, 0, -1)),  //   1.474, 0.265, a01450min
    Part_t(10111, ParticleInfo(0, 0, 2, 0, 0)),   //   1.474, 0.265, a01450zer
    Part_t(20333, ParticleInfo(0, 0, 0, 2, 0)),   //   1.4263, 0.0555, f11420zer
    Part_t(10311, ParticleInfo(0, 1, 1, 4, 0)),   //   1.4256, 0.0985, Ka1430zer
    Part_t(-10311,ParticleInfo(0, -1, 1, 4, 0)),  //   1.4256, 0.0985, Ka1430zrb
    Part_t(10321, ParticleInfo(0, 1, 1, 4, 1)),   //   1.4256, 0.0985, Ka1430plu
    Part_t(-10321,ParticleInfo(0, -1, 1, 4, -1)), //   1.4256, 0.0985, Ka1430min
    Part_t(20313, ParticleInfo(0, 1, 1, 2, 0)),   //   1.414, 0.232, Ka1410zer
    Part_t(-20313,ParticleInfo(0, -1, 1, 2, 0)),  //   1.414, 0.232, Ka1410zrb
    Part_t(20313, ParticleInfo(0, 1, 1, 2, 0)),   //   1.402, 0.174, Ka1400zer
    Part_t(-20313,ParticleInfo(0, -1, 1, 2, 0)),  //   1.402, 0.174, Ka1400zrb
    Part_t(20323, ParticleInfo(0, 1, 1, 2, 1)),   //   1.402, 0.174, Ka1400plu
    Part_t(-20323,ParticleInfo(0, -1, 1, 2, -1)), //   1.402, 0.174, Ka1400min
    Part_t(3114,  ParticleInfo(1, -1, 2, 3, -1)), //   1.3872, 0.0394, Sg1385min
    Part_t(-3114, ParticleInfo(-1, 1, 2, 3, 1)),  //   1.3872, 0.0394, Sg1385mnb
    Part_t(3214,  ParticleInfo(1, -1, 2, 3, 0)),  //   1.3837, 0.036, Sg1385zer
    Part_t(-3214, ParticleInfo(-1, 1, 2, 3, 0)),  //   1.3837, 0.036, Sg1385zrb
    Part_t(3224,  ParticleInfo(1, -1, 2, 3, 1)),  //   1.3828, 0.0358, Sg1385plu
    Part_t(-3224, ParticleInfo(-1, 1, 2, 3, -1)), //   1.3828, 0.0358, Sg1385plb
    Part_t(10221, ParticleInfo(0, 0, 0, 0, 0)),   //   1.35, 0.35, f01370zer
    Part_t(3312,  ParticleInfo(1, -2, 1, 1, -1)), //   1.32131, 4.009e-16, Xi1321min
    Part_t(-3312, ParticleInfo(-1, 2, 1, 1, 1)),  //   1.32131, 4.009e-16, Xi1321mnb
    Part_t(3322,  ParticleInfo(1, -2, 1, 1, 0)),  //   1.31483, 2.265e-16, Xi1321zer
    Part_t(-3322, ParticleInfo(-1, 2, 1, 1, 0)),  //   1.31483, 2.265e-16, Xi1321zrb
    Part_t(215,   ParticleInfo(0, 0, 2, 4, 1)),   //   1.318, 0.107, a21320plu
    Part_t(115,   ParticleInfo(0, 0, 2, 4, 0)),   //   1.318, 0.107, a21320zer
    Part_t(20223, ParticleInfo(0, 0, 0, 2, 0)),   //   1.2819, 0.024, f11285zer
    Part_t(225,   ParticleInfo(0, 0, 0, 4, 0)),   //   1.2754, 0.185, f21270zer
    Part_t(10313, ParticleInfo(0, 1, 1, 2, 0)),   //   1.273, 0.09, Ka1270zer
    Part_t(-10313,ParticleInfo(0, -1, 1, 2, 0)),  //   1.273, 0.09, Ka1270zrb
    Part_t(10323, ParticleInfo(0, 1, 1, 2, 1)),   //   1.273, 0.09, Ka1270plu
    Part_t(-10323,ParticleInfo(0, -1, 1, 2, -1)), //   1.273, 0.09, Ka1270min
    Part_t(2224,  ParticleInfo(1, 0, 3, 3, 2)),   //   1.232, 0.12, Dl1232plp
    Part_t(2214,  ParticleInfo(1, 0, 3, 3, 1)),   //   1.232, 0.12, Dl1232plu
    Part_t(2114,  ParticleInfo(1, 0, 3, 3, 0)),   //   1.232, 0.12, Dl1232zer
    Part_t(1114,  ParticleInfo(1, 0, 3, 3, -1)),  //   1.232, 0.12, Dl1232min
    Part_t(-2224, ParticleInfo(-1, 0, 3, 3, -2)), //   1.232, 0.12, Dl1232ppb
    Part_t(-2214, ParticleInfo(-1, 0, 3, 3, -1)), //   1.232, 0.12, Dl1232plb
    Part_t(-2114, ParticleInfo(-1, 0, 3, 3, 0)),  //   1.232, 0.12, Dl1232zrb
    Part_t(-1114, ParticleInfo(-1, 0, 3, 3, 1)),  //   1.232, 0.12, Dl1232mnb
    Part_t(20213, ParticleInfo(0, 0, 2, 2, 1)),   //   1.23, 0.425, a11260plu
    Part_t(-20213,ParticleInfo(0, 0, 2, 2, -1)),  //   1.23, 0.425, a11260min
    Part_t(20113, ParticleInfo(0, 0, 2, 2, 0)),   //   1.23, 0.425, a11260zer
    Part_t(10213, ParticleInfo(0, 0, 2, 2, 0)),   //   1.2295, 0.142, b11235zer
    Part_t(3112,  ParticleInfo(1, -1, 2, 1, -1)), //   1.197, 4.442e-16, Sg1189min
    Part_t(-3112, ParticleInfo(-1, 1, 2, 1, 1)),  //   1.197, 4.442e-16, Sg1189mnb
    Part_t(3212,  ParticleInfo(1, -1, 2, 1, 0)),  //   1.193, 8.879e-07, Sg1192zer
    Part_t(-3212, ParticleInfo(-1, 1, 2, 1, 0)),  //   1.193, 8.879e-07, Sg1192zrb
    Part_t(3222,  ParticleInfo(1, -1, 2, 1, 1)),  //   1.189, 8.195e-16, Sg1189plu
    Part_t(-3222, ParticleInfo(-1, 1, 2, 1, -1)), //   1.189, 8.195e-16, Sg1189plb
    Part_t(10223, ParticleInfo(0, 0, 0, 2, 0)),   //   1.17, 0.36, h11170zer
    Part_t(3122,  ParticleInfo(1, -1, 0, 1, 0)),  //   1.11568, 2.496e-16, Lm1115zer
    Part_t(-3122, ParticleInfo(-1, 1, 0, 1, 0)),  //   1.11568, 2.496e-16, Lm1115zrb
    Part_t(333,   ParticleInfo(0, 0, 0, 2, 0)),   //   1.01942, 0.004458, ph1020zer
    Part_t(331,   ParticleInfo(0, 0, 0, 0, 0)),   //   0.95778, 0.000202, eta0prime
    Part_t(2112,  ParticleInfo(1, 0, 1, 1, 0)),   //   0.939565, 0, ne0939zer
    Part_t(-2112, ParticleInfo(-1, 0, 1, 1, 0)),  //   0.939565, 0, ne0939zrb
    Part_t(2212,  ParticleInfo(1, 0, 1, 1, 1)),   //   0.938272, 0, pr0938plu
    Part_t(-2212, ParticleInfo(-1, 0, 1, 1, -1)), //   0.938272, 0, pr0938plb
    Part_t(313,   ParticleInfo(0, 1, 1, 2, 0)),   //   0.8961, 0.0507, Ka0892zer
    Part_t(-313,  ParticleInfo(0, -1, 1, 2, 0)),  //   0.8961, 0.0507, Ka0892zrb
    Part_t(323,   ParticleInfo(0, 1, 1, 2, 1)),   //   0.89166, 0.0508, Ka0892plu
    Part_t(-323,  ParticleInfo(0, -1, 1, 2, -1)), //   0.89166, 0.0508, Ka0892min
    Part_t(223,   ParticleInfo(0, 0, 0, 2, 0)),   //   0.78257, 0.00844, om0782zer
    Part_t(213,   ParticleInfo(0, 0, 2, 2, 1)),   //   0.7693, 0.1502, rho770plu
    Part_t(-213,  ParticleInfo(0, 0, 2, 2, -1)),  //   0.7693, 0.1502, rho770min
    Part_t(113,   ParticleInfo(0, 0, 2, 2, 0)),   //   0.7693, 0.1502, rho770zer
    Part_t(221,   ParticleInfo(0, 0, 0, 0, 0)),   //   0.5473, 1.29e-06, eta547zer
    Part_t(311,   ParticleInfo(0, 1, 1, 0, 0)),   //   0.497672, 7.335e-16, Ka0492zer
    Part_t(-311,  ParticleInfo(0, -1, 1, 0, 0)),  //   0.497672, 7.335e-16, Ka0492zrb
    Part_t(321,   ParticleInfo(0, 1, 1, 0, 1)),   //   0.493677, 0, Ka0492plu
    Part_t(-321,  ParticleInfo(0, -1, 1, 0, -1)), //   0.493677, 0, Ka0492min
    Part_t(211,   ParticleInfo(0, 0, 2, 0, 1)),   //   0.13957, 0, pi0139plu
    Part_t(-211,  ParticleInfo(0, 0, 2, 0, -1)),  //   0.13957, 0, pi0139min
    Part_t(111,   ParticleInfo(0, 0, 2, 0, 0)),   //   0.134976, 0, pi0135zer
    Part_t(22,    ParticleInfo(0, 0, 0, 2, 0)),   //   0.00, 0, gam000zer
  };

const std::map<const int, ParticleInfo> gParticleTable(gTableInitializer,
                                                       gTableInitializer + sizeof gTableInitializer / sizeof (Part_t));
  

//Part_t(443,  ParticleInfo(0, 0, 0, 2, 0)),// 3.09687, 0, jp3096zer
//Part_t(1231, ParticleInfo(0, 0, 1, 0, 0)),// 1.8693, 0, Dc1800plu
//Part_t(1232, ParticleInfo(0, 0, 1, 0, 0)),// 1.8693, 0, Dc1800min
//Part_t(1233, ParticleInfo(0, 0, 1, 0, 0)), //1.8693, 0, Dc1800zer
//Part_t(1234, ParticleInfo(0, 0, 1, 0, 0)), //1.8693, 0, Dc1800zrb
//Part_t(4231, ParticleInfo(0, 0, 1, 0, 0)), //2.01, 0, Dc2010plu
//Part_t(4232, ParticleInfo(0, 0, 1, 0, 0)), //2.0103, 0, Dc2010min
//Part_t(4233, ParticleInfo(0, 0, 1, 0, 0)), //2.0103, 0, Dc2010zer
//Part_t(4234, ParticleInfo(0, 0, 1, 0, 0)), //2.0103, 0, Dc2010zrb
//Part_t(9401, ParticleInfo(1, 0, 1, 11, 1)),// 2.6, 0.65, Ns2600plu
//Part_t(9400, ParticleInfo(1, 0, 1, 11, 0)), //2.6, 0.65, Ns2600zer
//Part_t(-9401,ParticleInfo( -1, 0, 1, 11, -1)), //2.6, 0.65, Ns2600plb
//Part_t(-9400,ParticleInfo( -1, 0, 1, 11, 0)),// 2.6, 0.65, Ns2600zrb
//Part_t(9297, ParticleInfo(1, 0, 3, 11, 2)), //2.42, 0.4, Dl2420plp
//Part_t(9298, ParticleInfo(1, 0, 3, 11, 1)), //2.42, 0.4, Dl2420plu
//Part_t(9299, ParticleInfo(1, 0, 3, 11, 0)), //2.42, 0.4, Dl2420zer
//Part_t(9300, ParticleInfo(1, 0, 3, 11, -1)), //2.42, 0.4, Dl2420min
//Part_t(-9297,ParticleInfo( -1, 0, 3, 11, -2)), //2.42, 0.4, Dl2420ppb
//Part_t(-9298,ParticleInfo( -1, 0, 3, 11, -1)), //2.42, 0.4, Dl2420plb
//Part_t(-9299,ParticleInfo( -1, 0, 3, 11, 0)), //2.42, 0.4, Dl2420zrb
//Part_t(-9300,ParticleInfo( -1, 0, 3, 11, 1)),// 2.42, 0.4, Dl2420mnb
//Part_t(9001, ParticleInfo(1, -1, 0, 3, 0)), //2.35, 0.15, Lm2350zer
//Part_t(-9001,ParticleInfo( -1, 1, 0, 3, 0)), //2.35, 0.15, Lm2350zrb
//Part_t(40225,ParticleInfo( 0, 0, 0, 8, 0)), //2.339, 0.319, f42340zer
//Part_t(30225,ParticleInfo( 0, 0, 0, 4, 0)), //2.297, 0.149, f22300zer
//Part_t(9000, ParticleInfo(1, -3, 0, 3, -1)),// 2.252, 0.055, UM2250min
//Part_t(-9000,ParticleInfo( -1, 3, 0, 3, 1)),// 2.252, 0.055, UM2250mnb
//Part_t(5128, ParticleInfo(1, 0, 1, 9, 1)), //2.25, 0.4, Ns2250plu
//Part_t(5218, ParticleInfo(1, 0, 1, 9, 0)), //2.25, 0.4, Ns2250zer
//Part_t(-5128,ParticleInfo( -1, 0, 1, 9, -1)),// 2.25, 0.4, Ns2250plb
//Part_t(-5218,ParticleInfo( -1, 0, 1, 9, 0)), //2.25, 0.4, Ns2250zrb
//Part_t(4028, ParticleInfo(1, -1, 2, 3, 1)), //2.25, 0.1, Sg2250plu
//Part_t(4128, ParticleInfo(1, -1, 2, 3, -1)), //2.25, 0.1, Sg2250min
//Part_t(4228, ParticleInfo(1, -1, 2, 3, 0)), //2.25, 0.1, Sg2250zer
//Part_t(-4028,ParticleInfo( -1, 1, 2, 3, -1)), //2.25, 0.1, Sg2250plb
//Part_t(-4128,ParticleInfo( -1, 1, 2, 3, 1)), //2.25, 0.1, Sg2250mnb
//Part_t(-4228,ParticleInfo( -1, 1, 2, 3, 0)), //2.25, 0.1, Sg2250zrb
//Part_t(3128, ParticleInfo(1, 0, 1, 5, 1)),// 2.22, 0.4, Ns2220plu
//Part_t(3218, ParticleInfo(1, 0, 1, 5, 0)),// 2.22, 0.4, Ns2220zer
//Part_t(-3128,ParticleInfo( -1, 0, 1, 5, -1)),// 2.22, 0.4, Ns2220plb
//Part_t(-3218,ParticleInfo( -1, 0, 1, 5, 0)),// 2.22, 0.4, Ns2220zrb
//Part_t(1218, ParticleInfo(1, 0, 1, 7, 1)), //2.19, 0.45, Ns2190plu
//Part_t(2128, ParticleInfo(1, 0, 1, 7, 0)), //2.19, 0.45, Ns2190zer
//Part_t(-1218,ParticleInfo( -1, 0, 1, 7, -1)), //2.19, 0.45, Ns2190plb
//Part_t(-2128,ParticleInfo( -1, 0, 1, 7, 0)), //2.19, 0.45, Ns2190zrb
//Part_t(23126,ParticleInfo( 1, -1, 0, 3, 0)), //2.11, 0.2, Lm2110zer
//Part_t(-23126,ParticleInfo( -1, 1, 0, 3, 0)), //2.11, 0.2, Lm2110zrb
//Part_t(3128, ParticleInfo(1, -1, 0, 3, 0)), //2.1, 0.2, Lm2100zer
//Part_t(-3128,ParticleInfo( -1, 1, 0, 3, 0)), //2.1, 0.2, Lm2100zrb
//Part_t(329,  ParticleInfo( 0, 1, 1, 4, 1)), //2.045, 0.198, Ka2045plu
//Part_t(-329, ParticleInfo(0, -1, 1, 4, -1)),// 2.045, 0.198, Ka2045min
//Part_t(319,  ParticleInfo(0, 1, 1, 4, 0)),// 2.045, 0.198, Ka2045zer
//Part_t(-319, ParticleInfo(0, -1, 1, 4, 0)),// 2.045, 0.198, Ka2045zrb
//Part_t(229,  ParticleInfo(0, 0, 0, 8, 0)), //2.034, 0.222, f42050zer
//Part_t(3118, ParticleInfo(1, -1, 2, 3, 1)),// 2.03, 0.18, Sg2030plu
//Part_t(3218, ParticleInfo(1, -1, 2, 3, -1)),// 2.03, 0.18, Sg2030min
//Part_t(3228, ParticleInfo(1, -1, 2, 3, 0)), //2.03, 0.18, Sg2030zer
//Part_t(-3118,ParticleInfo( -1, 1, 2, 3, -1)), //2.03, 0.18, Sg2030plb
//Part_t(-3218,ParticleInfo( -1, 1, 2, 3, 1)), //2.03, 0.18, Sg2030mnb
//Part_t(-3228,ParticleInfo( -1, 1, 2, 3, 0)), //2.03, 0.18, Sg2030zrb
//Part_t(8901, ParticleInfo(1, -2, 1, 5, -1)), //2.025, 0.02, Xi2030min
//Part_t(8900, ParticleInfo(-1, 2, 1, 5, 1)), //2.025, 0.02, Xi2030mnb
//Part_t(-8901,ParticleInfo( 1, -2, 1, 5, 0)), //2.025, 0.02, Xi2030zer
//Part_t(-8900,ParticleInfo( -1, 2, 1, 5, 0)), //2.025, 0.02, Xi2030zrb
//Part_t(219,  ParticleInfo(0, 0, 2, 8, 1)), //2.014, 0.361, a42040plu
//Part_t(-219, ParticleInfo(0, 0, 2, 8, -1)), //2.014, 0.361, a42040min
//Part_t(119,  ParticleInfo(0, 0, 2, 8, 0)), //2.014, 0.361, a42040zer
//Part_t(20225,ParticleInfo( 0, 0, 0, 4, 0)), //2.011, 0.202, f22010zer
//Part_t(1118, ParticleInfo(1, 0, 3, 7, 2)), //1.95, 0.3, Dl1950plp
//Part_t(2118, ParticleInfo(1, 0, 3, 7, 1)), //1.95, 0.3, Dl1950plu
//Part_t(2218, ParticleInfo(1, 0, 3, 7, 0)), //1.95, 0.3, Dl1950zer
//Part_t(2228, ParticleInfo(1, 0, 3, 7, -1)),// 1.95, 0.3, Dl1950min
//Part_t(-1118,ParticleInfo( -1, 0, 3, 7, -2)), //1.95, 0.3, Dl1950ppb
//Part_t(-2118,ParticleInfo( -1, 0, 3, 7, -1)), //1.95, 0.3, Dl1950plb
//Part_t(-2218,ParticleInfo( -1, 0, 3, 7, 0)), //1.95, 0.3, Dl1950zrb
//Part_t(-2228,ParticleInfo( -1, 0, 3, 7, 1)), //1.95, 0.3, Dl1950mnb
//Part_t(67001,ParticleInfo( 1, -2, 1, 3, -1)), //1.95, 0.06, Xi1950min
//Part_t(-67001,ParticleInfo( -1, 2, 1, 3, 1)), //1.95, 0.06, Xi1950mnb
//Part_t(67000,ParticleInfo(1, -2, 1, 3, 0)), //1.95, 0.06, Xi1950zer
//Part_t(-67000,ParticleInfo( -1, 2, 1, 3, 0)), //1.95, 0.06, Xi1950zrb
//Part_t(23114, ParticleInfo(1, -1, 2, 3, 1)), //1.94, 0.22, Sg1940plu
//Part_t(23214, ParticleInfo(1, -1, 2, 3, -1)), //1.94, 0.22, Sg1940min
//Part_t(23224, ParticleInfo(1, -1, 2, 3, 0)), //1.94, 0.22, Sg1940zer
//Part_t(-23114,ParticleInfo( -1, 1, 2, 3, -1)), //1.94, 0.22, Sg1940plb
//Part_t(-23214,ParticleInfo( -1, 1, 2, 3, 1)), //1.94, 0.22, Sg1940mnb
//Part_t(-23224,ParticleInfo(-1, 1, 2, 3, 0)), //1.94, 0.22, Sg1940zrb
//Part_t(11116, ParticleInfo(1, 0, 3, 5, 2)), //1.93, 0.35, Dl1930plp
//Part_t(11216, ParticleInfo(1, 0, 3, 5, 1)), //1.93, 0.35, Dl1930plu
//Part_t(12126, ParticleInfo(1, 0, 3, 5, 0)), //1.93, 0.35, Dl1930zer
//Part_t(12226, ParticleInfo(1, 0, 3, 5, -1)), //1.93, 0.35, Dl1930min
//Part_t(-11116,ParticleInfo( -1, 0, 3, 5, -2)),// 1.93, 0.35, Dl1930ppb
//Part_t(-11216,ParticleInfo( -1, 0, 3, 5, -1)),// 1.93, 0.35, Dl1930plb
//Part_t(-12126,ParticleInfo( -1, 0, 3, 5, 0)), //1.93, 0.35, Dl1930zrb
//Part_t(-12226,ParticleInfo( -1, 0, 3, 5, 1)),// 1.93, 0.35, Dl1930mnb
//Part_t(21114, ParticleInfo(1, 0, 3, 3, 2)), //1.92, 0.2, Dl1920plp
//Part_t(22114, ParticleInfo(1, 0, 3, 3, 1)), //1.92, 0.2, Dl1920plu
//Part_t(22214, ParticleInfo(1, 0, 3, 3, 0)), //1.92, 0.2, Dl1920zer
//Part_t(22224, ParticleInfo(1, 0, 3, 3, -1)), //1.92, 0.2, Dl1920min
//Part_t(-21114,ParticleInfo( -1, 0, 3, 3, -2)), //1.92, 0.2, Dl1920ppb
//Part_t(-22114,ParticleInfo( -1, 0, 3, 3, -1)), //1.92, 0.2, Dl1920plb
//Part_t(-22214,ParticleInfo( -1, 0, 3, 3, 0)), //1.92, 0.2, Dl1920zrb
//Part_t(-22224,ParticleInfo( -1, 0, 3, 3, 1)), //1.92, 0.2, Dl1920mnb
//Part_t(13116, ParticleInfo(1, -1, 2, 5, 1)), //1.915, 0.12, Sg1915plu
//Part_t(13216, ParticleInfo(1, -1, 2, 5, -1)), //1.915, 0.12, Sg1915min
//Part_t(13226, ParticleInfo(1, -1, 2, 5, 0)), //1.915, 0.12, Sg1915zer
//Part_t(-13116,ParticleInfo( -1, 1, 2, 5, -1)), //1.915, 0.12, Sg1915plb
//Part_t(-13216,ParticleInfo( -1, 1, 2, 5, 1)), //1.915, 0.12, Sg1915mnb
//Part_t(-13226,ParticleInfo( -1, 1, 2, 5, 0)), //1.915, 0.12, Sg1915zrb
//Part_t(21112, ParticleInfo(1, 0, 3, 1, 2)),// 1.91, 0.25, Dl1910plp
//Part_t(21212, ParticleInfo(1, 0, 3, 1, 1)),// 1.91, 0.25, Dl1910plu
//Part_t(22122, ParticleInfo(1, 0, 3, 1, 0)),// 1.91, 0.25, Dl1910zer
//Part_t(22222, ParticleInfo(1, 0, 3, 1, -1)), //1.91, 0.25, Dl1910min
//Part_t(-21112,ParticleInfo( -1, 0, 3, 1, -2)),// 1.91, 0.25, Dl1910ppb
//Part_t(-21212,ParticleInfo( -1, 0, 3, 1, -1)), //1.91, 0.25, Dl1910plb
//Part_t(-22122,ParticleInfo( -1, 0, 3, 1, 0)), //1.91, 0.25, Dl1910zrb
//Part_t(-22222,ParticleInfo( -1, 0, 3, 1, 1)), //1.91, 0.25, Dl1910mnb
//Part_t(1116,  ParticleInfo(1, 0, 3, 5, 2)), //1.905, 0.35, Dl1905plp
//Part_t(1216,  ParticleInfo(1, 0, 3, 5, 1)), //1.905, 0.35, Dl1905plu
//Part_t(2126,  ParticleInfo(1, 0, 3, 5, 0)), //1.905, 0.35, Dl1905zer
//Part_t(2226,  ParticleInfo(1, 0, 3, 5, -1)), //1.905, 0.35, Dl1905min
//Part_t(-1116, ParticleInfo(-1, 0, 3, 5, -2)),// 1.905, 0.35, Dl1905ppb
//Part_t(-1216, ParticleInfo(-1, 0, 3, 5, -1)),// 1.905, 0.35, Dl1905plb
//Part_t(-2126, ParticleInfo(-1, 0, 3, 5, 0)), //1.905, 0.35, Dl1905zrb
//Part_t(-2226, ParticleInfo(-1, 0, 3, 5, 1)), //1.905, 0.35, Dl1905mnb
//Part_t(23124, ParticleInfo(1, -1, 0, 3, 0)), //1.89, 0.1, Lm1890zer
//Part_t(-23124,ParticleInfo( -1, 1, 0, 3, 0)),// 1.89, 0.1, Lm1890zrb
//Part_t(-80000,ParticleInfo( 2, 0, 4, 0, 1)),// 1.87561, 0, de2000plb
//Part_t(80000, ParticleInfo(2, 0, 4, 0, 1)), //1.87561, 0, de2000plu
//Part_t(337,   ParticleInfo(0, 0, 0, 6, 0)), //1.854, 0.087, ph1850zer
//Part_t(13126, ParticleInfo(1, -1, 0, 1, 0)), //1.83, 0.95, Lm1830zer
//Part_t(-13126,ParticleInfo( -1, 1, 0, 1, 0)), //1.83, 0.95, Lm1830zrb
//Part_t(13314, ParticleInfo(1, -2, 1, 3, -1)), //1.823, 0.024, Xi1820min
//Part_t(13324, ParticleInfo(-1, 2, 1, 3, 1)), //1.823, 0.024, Xi1820mnb
//Part_t(-13314,ParticleInfo( 1, -2, 1, 3, 0)), //1.823, 0.024, Xi1820zer
//Part_t(-13324,ParticleInfo( -1, 2, 1, 3, 0)), //1.823, 0.024, Xi1820zrb
//Part_t(3126,  ParticleInfo(1, -1, 0, 1, 0)), //1.82, 0.8, Lm1820zer
//Part_t(-3126, ParticleInfo(-1, 1, 0, 1, 0)), //1.82, 0.8, Lm1820zrb
//Part_t(20315, ParticleInfo(0, 1, 1, 4, 1)), //1.816, 0.276, Ka1820plu
//Part_t(20325, ParticleInfo(0, -1, 1, 4, -1)), //1.816, 0.276, Ka1820min
//Part_t(-20325,ParticleInfo( 0, 1, 1, 4, 0)), //1.816, 0.276, Ka1820zer
//Part_t(-20315,ParticleInfo( 0, -1, 1, 4, 0)), //1.816, 0.276, Ka1820zrb
//Part_t(53122, ParticleInfo(1, -1, 0, 1, 0)), //1.81, 0.15, Lm1810zer
//Part_t(-53122,ParticleInfo( -1, 1, 0, 1, 0)),// 1.81, 0.15, Lm1810zrb
//Part_t(200111,ParticleInfo( 0, 0, 2, 0, 1)), //1.801, 0.21, pi1800plu
//Part_t(-200111,ParticleInfo( 0, 0, 2, 0, -1)),// 1.801, 0.21, pi1800min
//Part_t(200211, ParticleInfo(0, 0, 2, 0, 0)), //1.801, 0.21, pi1800zer
//Part_t(43122, ParticleInfo(1, -1, 0, 1, 0)), //1.8, 0.3, Lm1800zer
//Part_t(-43122, ParticleInfo(-1, 1, 0, 1, 0)),// 1.8, 0.3, Lm1800zrb
//Part_t(317,   ParticleInfo(0, 1, 1, 6, 0)), //1.776, 0.159, Ka1780zer
//Part_t(327,   ParticleInfo(0, -1, 1, 6, 0)), //1.776, 0.159, Ka1780zrb
//Part_t(-317,  ParticleInfo(0, 1, 1, 6, 1)), //1.776, 0.159, Ka1780plu
//Part_t(-327,  ParticleInfo(0, -1, 1, 6, -1)),// 1.776, 0.159, Ka1780min
//Part_t(3116,  ParticleInfo(1, -1, 2, 5, 1)), //1.775, 0.12, Sg1775plu
//Part_t(3216,  ParticleInfo(1, -1, 2, 5, -1)),// 1.775, 0.12, Sg1775min
//Part_t(3226,  ParticleInfo(1, -1, 2, 5, 0)), //1.775, 0.12, Sg1775zer
//Part_t(-3116, ParticleInfo(-1, 1, 2, 5, -1)),// 1.775, 0.12, Sg1775plb
//Part_t(-3216, ParticleInfo(-1, 1, 2, 5, 1)), //1.775, 0.12, Sg1775mnb
//Part_t(-3226, ParticleInfo(-1, 1, 2, 5, 0)), //1.775, 0.12, Sg1775zrb
//Part_t(8116,  ParticleInfo(1, -1, 2, 1, 1)), //1.75, 0.09, Sg1750plu
//Part_t(8117,  ParticleInfo(1, -1, 2, 1, 0)), //1.75, 0.09, Sg1750zer
//Part_t(8118,  ParticleInfo(1, -1, 2, 1, -1)), //1.75, 0.09, Sg1750min
//Part_t(-8116, ParticleInfo(-1, 1, 2, 1, -1)), //1.75, 0.09, Sg1750plb
//Part_t(-8117, ParticleInfo(-1, 1, 2, 1, 0)), //1.75, 0.09, Sg1750zrb
//Part_t(-8118, ParticleInfo(-1, 1, 2, 1, 1)), //1.75, 0.09, Sg1750mnb
//Part_t(10315, ParticleInfo(0, 1, 1, 4, 0)), //1.773, 0.186, Ka1770zer
//Part_t(10325, ParticleInfo(0, -1, 1, 4, 0)),// 1.773, 0.186, Ka1770zrb
//Part_t(-10315,ParticleInfo( 0, 1, 1, 4, 1)), //1.773, 0.186, Ka1770plu
//Part_t(-10325,ParticleInfo( 0, -1, 1, 4, -1)),// 1.773, 0.186, Ka1770min
//Part_t(31214, ParticleInfo(1, 0, 1, 3, 1)), //1.72, 0.15, Ns1720plu
//Part_t(32124, ParticleInfo(1, 0, 1, 3, 0)), //1.72, 0.15, Ns1720zer
//Part_t(-31214,ParticleInfo( -1, 0, 1, 3, -1)), //1.72, 0.15, Ns1720plb
//Part_t(-32124,ParticleInfo( -1, 0, 1, 3, 0)), //1.72, 0.15, Ns1720zrb
//Part_t(30313, ParticleInfo(0, 1, 1, 2, 0)), //1.717, 0.322, Ka1680zer
//Part_t(30323, ParticleInfo(0, -1, 1, 2, 0)), //1.717, 0.322, Ka1680zrb
//Part_t(-30313,ParticleInfo( 0, 1, 1, 2, 1)), //1.717, 0.322, Ka1680plu
//Part_t(-30323,ParticleInfo( 0, -1, 1, 2, -1)),// 1.717, 0.322, Ka1680min
//Part_t(42112, ParticleInfo(1, 0, 1, 1, 1)), //1.71, 0.1, Ns1710plu
//Part_t(42212, ParticleInfo(1, 0, 1, 1, 0)), //1.71, 0.1, Ns1710zer
//Part_t(-42112,ParticleInfo( -1, 0, 1, 1, -1)),// 1.71, 0.1, Ns1710plb
//Part_t(-42212,ParticleInfo( -1, 0, 1, 1, 0)), //1.71, 0.1, Ns1710zrb
//Part_t(21214, ParticleInfo(1, 0, 1, 3, 1)), //1.7, 0.1, Ns1700plu
//Part_t(22124, ParticleInfo(1, 0, 1, 3, 0)),// 1.7, 0.1, Ns1700zer
//Part_t(-21214,ParticleInfo( -1, 0, 1, 3, -1)),// 1.7, 0.1, Ns1700plb
//Part_t(-22124,ParticleInfo( -1, 0, 1, 3, 0)),// 1.7, 0.1, Ns1700zrb
//Part_t(30213, ParticleInfo(0, 0, 2, 2, 1)), //1.7, 0.24, rh1700plu
//Part_t(-30213,ParticleInfo( 0, 0, 2, 2, -1)), //1.7, 0.24, rh1700min
//Part_t(30113, ParticleInfo(0, 0, 2, 2, 0)), //1.7, 0.24, rh1700zer
//Part_t(11114, ParticleInfo(1, 0, 3, 3, 2)), //1.7, 0.3, Dl1700plp
//Part_t(12114, ParticleInfo(1, 0, 3, 3, 1)), //1.7, 0.3, Dl1700plu
//Part_t(12214, ParticleInfo(1, 0, 3, 3, 0)), //1.7, 0.3, Dl1700zer
//Part_t(12224, ParticleInfo(1, 0, 3, 3, -1)), //1.7, 0.3, Dl1700min
//Part_t(-11114,ParticleInfo( -1, 0, 3, 3, -2)), //1.7, 0.3, Dl1700ppb
//Part_t(-12114,ParticleInfo( -1, 0, 3, 3, -1)), //1.7, 0.3, Dl1700plb
//Part_t(-12214,ParticleInfo( -1, 0, 3, 3, 0)), //1.7, 0.3, Dl1700zrb
//Part_t(-12224,ParticleInfo( -1, 0, 3, 3, 1)), //1.7, 0.3, Dl1700mnb
//Part_t(217,   ParticleInfo(0, 0, 2, 6, 1)), //1.691, 0.161, rh1690plu
//Part_t(-217,  ParticleInfo(0, 0, 2, 6, -1)), //1.691, 0.161, rh1690min
//Part_t(117,   ParticleInfo(0, 0, 2, 0, 0)), //1.691, 0.161, rh1690zer
//Part_t(13124, ParticleInfo(1, -1, 0, 3, 0)), //1.69, 0.06, Lm1690zer
//Part_t(-13124,ParticleInfo( -1, 1, 0, 3, 0)), //1.69, 0.06, Lm1690zrb
//Part_t(-67719,ParticleInfo(1, -2, 1, 3, -1)), //1.69, 0.029, Xi1690min
//Part_t(67719, ParticleInfo(-1, 2, 1, 3, 1)), //1.69, 0.029, Xi1690mnb
//Part_t(67718, ParticleInfo(1, -2, 1, 3, 0)), //1.69, 0.029, Xi1690zer
//Part_t(-67718,ParticleInfo(-1, 2, 1, 3, 0)), //1.69, 0.029, Xi1690zrb
//Part_t(12116, ParticleInfo(1, 0, 1, 5, 1)), //1.68, 0.13, Ns1680plu
//Part_t(12216, ParticleInfo(1, 0, 1, 5, 0)), //1.68, 0.13, Ns1680zer
//Part_t(-12116,ParticleInfo( -1, 0, 1, 5, -1)),// 1.68, 0.13, Ns1680plb
//Part_t(-12216,ParticleInfo( -1, 0, 1, 5, 0)), //1.68, 0.13, Ns1680zrb
//Part_t(100333,ParticleInfo( 0, 0, 0, 2, 0)), //1.68, 0.15, ph1680zer
//Part_t(2116,  ParticleInfo(1, 0, 1, 5, 1)), //1.675, 0.15, Ns1675plu
//Part_t(2216,  ParticleInfo(1, 0, 1, 5, 0)), //1.675, 0.15, Ns1675zer
//Part_t(-2116, ParticleInfo(-1, 0, 1, 5, -1)), //1.675, 0.15, Ns1675plb
//Part_t(-2216, ParticleInfo(-1, 0, 1, 5, 0)), //1.675, 0.15, Ns1675zrb
//Part_t(10215, ParticleInfo(0, 0, 2, 4, 1)), //1.67, 0.259, pi1670plu
//Part_t(-10215,ParticleInfo( 0, 0, 2, 4, -1)), //1.67, 0.259, pi1670min
//Part_t(10115, ParticleInfo(0, 0, 2, 4, 0)), //1.67, 0.259, pi1670zer
//Part_t(33122, ParticleInfo(1, -1, 0, 1, 0)), //1.67, 0.035, Lm1670zer
//Part_t(-33122,ParticleInfo( -1, 1, 0, 1, 0)), //1.67, 0.035, Lm1670zrb
//Part_t(13114, ParticleInfo(1, -1, 2, 3, 1)), //1.67, 0.06, Sg1670plu
//Part_t(13214, ParticleInfo(1, -1, 2, 3, -1)),// 1.67, 0.06, Sg1670min
//Part_t(13224, ParticleInfo(1, -1, 2, 3, 0)), //1.67, 0.06, Sg1670zer
//Part_t(-13114,ParticleInfo( -1, 1, 2, 3, -1)), //1.67, 0.06, Sg1670plb
//Part_t(-13214,ParticleInfo( -1, 1, 2, 3, 1)), //1.67, 0.06, Sg1670mnb
//Part_t(-13224,ParticleInfo( -1, 1, 2, 3, 0)), //1.67, 0.06, Sg1670zrb
//Part_t(227,   ParticleInfo( 0, 0, 0, 6, 0)), //1.667, 0.168, om1670zer
//Part_t(13112, ParticleInfo(1, -1, 2, 1, 1)), //1.66, 0.1, Sg1660plu
//Part_t(13212, ParticleInfo(1, -1, 2, 1, -1)), //1.66, 0.1, Sg1660min
//Part_t(13222, ParticleInfo(1, -1, 2, 1, 0)), //1.66, 0.1, Sg1660zer
//Part_t(-13112,ParticleInfo( -1, 1, 2, 1, -1)),// 1.66, 0.1, Sg1660plb
//Part_t(-13212,ParticleInfo( -1, 1, 2, 1, 1)),// 1.66, 0.1, Sg1660mnb
//Part_t(-13222,ParticleInfo( -1, 1, 2, 1, 0)),// 1.66, 0.1, Sg1660zrb
//Part_t(32112, ParticleInfo(1, 0, 1, 1, 1)), //1.65, 0.15, Ns1650plu
//Part_t(32212, ParticleInfo(1, 0, 1, 1, 0)), //1.65, 0.15, Ns1650zer
//Part_t(-32112,ParticleInfo( -1, 0, 1, 1, -1)),// 1.65, 0.15, Ns1650plb
//Part_t(-32212,ParticleInfo( -1, 0, 1, 1, 0)),// 1.65, 0.15, Ns1650zrb
//Part_t(30223, ParticleInfo(0, 0, 0, 2, 0)), //1.649, 0.22, om1650zer
//Part_t(1112,  ParticleInfo(1, 0, 3, 1, 2)), //1.62, 0.15, Dl1620plp
//Part_t(1212,  ParticleInfo(1, 0, 3, 1, 1)), //1.62, 0.15, Dl1620plu
//Part_t(2122,  ParticleInfo(1, 0, 3, 1, 0)), //1.62, 0.15, Dl1620zer
//Part_t(2222,  ParticleInfo(1, 0, 3, 1, -1)), //1.62, 0.15, Dl1620min
//Part_t(-1112, ParticleInfo(-1, 0, 3, 1, -2)),// 1.62, 0.15, Dl1620ppb
//Part_t(-1212, ParticleInfo(-1, 0, 3, 1, -1)),// 1.62, 0.15, Dl1620plb
//Part_t(-2122, ParticleInfo(-1, 0, 3, 1, 0)), //1.62, 0.15, Dl1620zrb
//Part_t(-2222, ParticleInfo(-1, 0, 3, 1, 1)), //1.62, 0.15, Dl1620mnb
//Part_t(46653, ParticleInfo(1, -2, 1, 3, -1)), //1.62, 0.03, Xi1620min
//Part_t(-46653,ParticleInfo( -1, 2, 1, 3, 1)), //1.62, 0.03, Xi1620mnb
//Part_t(45553, ParticleInfo(1, -2, 1, 3, 0)), //1.62, 0.03, Xi1620zer
//Part_t(-45553,ParticleInfo( -1, 2, 1, 3, 0)),// 1.62, 0.03, Xi1620zrb
//Part_t(31114, ParticleInfo(1, 0, 3, 3, 2)), //1.6, 0.35, Dl1600plp
//Part_t(32114, ParticleInfo(1, 0, 3, 3, 1)), //1.6, 0.35, Dl1600plu
//Part_t(32214, ParticleInfo(1, 0, 3, 3, 0)), //1.6, 0.35, Dl1600zer
//Part_t(32224, ParticleInfo(1, 0, 3, 3, -1)), //1.6, 0.35, Dl1600min
//Part_t(-31114,ParticleInfo( -1, 0, 3, 3, -2)), //1.6, 0.35, Dl1600ppb
//Part_t(-32114,ParticleInfo( -1, 0, 3, 3, -1)), //1.6, 0.35, Dl1600plb
//Part_t(-32214,ParticleInfo( -1, 0, 3, 3, 0)), //1.6, 0.35, Dl1600zrb
//Part_t(-32224,ParticleInfo( -1, 0, 3, 3, 1)), //1.6, 0.35, Dl1600mnb
//Part_t(23122, ParticleInfo(1, -1, 0, 1, 0)), //1.6, 0.15, Lm1600zer
//Part_t(-23122,ParticleInfo( -1, 1, 0, 1, 0)),// 1.6, 0.15, Lm1600zrb
//Part_t(22212, ParticleInfo(1, 0, 1, 1, 1)), //1.535, 0.15, Ns1535plu
//Part_t(22122, ParticleInfo(1, 0, 1, 1, 0)), //1.535, 0.15, Ns1535zer
//Part_t(-22212,ParticleInfo( -1, 0, 1, 1, -1)), //1.535, 0.15, Ns1535plb
//Part_t(-22122,ParticleInfo( -1, 0, 1, 1, 0)), //1.535, 0.15, Ns1535zrb
//Part_t(2124,  ParticleInfo(1, 0, 1, 3, 1)), //1.52, 0.12, Ns1520plu
//Part_t(1214,  ParticleInfo(1, 0, 1, 3, 0)), //1.52, 0.12, Ns1520zer
//Part_t(-2124, ParticleInfo(-1, 0, 1, 3, -1)),// 1.52, 0.12, Ns1520plb
//Part_t(-1214, ParticleInfo(-1, 0, 1, 3, 0)), //1.52, 0.12, Ns1520zrb
//Part_t(3124,  ParticleInfo(1, -1, 0, 3, 0)), //1.5195, 0.0156, Lm1520zer
//Part_t(-3124, ParticleInfo(-1, 1, 0, 3, 0)), //1.5195, 0.0156, Lm1520zrb
//Part_t(9000223,ParticleInfo( 0, 0, 0, 0, 0)), //1.507, 0.112, f01500zer
//Part_t(100213,ParticleInfo( 0, 0, 2, 2, 1)),// 1.465, 0.31, rh1450plu
//Part_t(-100213,ParticleInfo( 0, 0, 2, 2, -1)), //1.465, 0.31, rh1450min
//Part_t(100113, ParticleInfo(0, 0, 2, 2, 0)), //1.465, 0.31, rh1450zer
//Part_t(12212, ParticleInfo(1, 0, 1, 1, 1)), //1.44, 0.35, Ns1440plu
//Part_t(12112, ParticleInfo(1, 0, 1, 1, 0)), //1.44, 0.35, Ns1440zer
//Part_t(-12212,ParticleInfo( -1, 0, 1, 1, -1)),// 1.44, 0.35, Ns1440plb
//Part_t(-12112,ParticleInfo( -1, 0, 1, 1, 0)),// 1.44, 0.35, Ns1440zrb
//Part_t(100331,ParticleInfo( 0, 0, 0, 0, 0)), //1.435, 0.065, et1440zer
//Part_t(100223,ParticleInfo( 0, 0, 0, 2, 0)), //1.419, 0.174, om1420zer
//Part_t(100323,ParticleInfo( 0, 1, 1, 2, 1)), //1.414, 0.232, Ka1410plu
//Part_t(-100323,ParticleInfo( 0, -1, 1, 2, -1)), //1.414, 0.232, Ka1410min
//Part_t(100313, ParticleInfo(0, 1, 1, 0, 0)), //1.412, 0.294, Ka1412zer
//Part_t(-100313, ParticleInfo(0, -1, 1, 0, 0)), //1.412, 0.294, Ka1412zrb
//Part_t(13122, ParticleInfo(1, -1, 0, 1, 0)), //1.406, 0.05, Lm1405zer
//Part_t(-13122,ParticleInfo(-1, 1, 0, 1, 0)), //1.406, 0.05, Lm1405zrb
//Part_t(100211, ParticleInfo(0, 0, 2, 0, 1)),// 1.3, 0.4, pi1300plu
//Part_t(-100211, ParticleInfo(0, 0, 2, 0, -1)), //1.3, 0.4, pi1300min
//Part_t(100111, ParticleInfo(0, 0, 2, 0, 0)), //1.3, 0.4, pi1300zer
//Part_t(100221, ParticleInfo(0, 0, 0, 0, 0)), //1.297, 0.053, et1295zer
//Part_t(9000211, ParticleInfo(0, 0, 2, 0, 1)), //0.9848, 0.075, a00980plu
//Part_t(-9000211, ParticleInfo(0, 0, 2, 0, -1)),// 0.9848, 0.075, a00980min
//Part_t(9000111, ParticleInfo(0, 0, 2, 0, 0)),// 0.9848, 0.075, a00980zer
//Part_t(9010221, ParticleInfo(0, 0, 0, 0, 0)),// 0.98, 0.1, f00980zer
//Part_t(9000221, ParticleInfo(0, 0, 0, 0, 0)), //0.8, 0.8, f00600zer

