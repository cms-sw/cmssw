import FWCore.ParameterSet.Config as cms

ecalMustacheSCParametersESProducer = cms.ESProducer("EcalMustacheSCParametersESProducer",
    sqrtLogClustETuning = cms.double(1.1),

    # Parameters from the analysis by L. Zygala [https://indico.cern.ch/event/949294/contributions/3988389/attachments/2091573/3514649/2020_08_26_Clustering.pdf]
    # mustache parambola parameters depending on cluster energy and seed crystal eta
    parabolaParameterSets = cms.VPSet(
        ## average parameters
        #cms.PSet(
        #    log10EMin = cms.double(-3.),
        #    etaMin = cms.double(0.),
        #    pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
        #    pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
        #    w0Up = cms.vdouble( -0.00681785, -0.00239516),
        #    w1Up = cms.vdouble( 0.000699995, -0.00554331),
        #    w0Low = cms.vdouble(-0.00681785, -0.00239516),
        #    w1Low = cms.vdouble(0.000699995, -0.00554331)
        #)
        # log10(E) and eta binned parameters
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.0242133, -0.00973345),
            w1Up = cms.vdouble(-0.0442618, -0.00926366),
            w0Low = cms.vdouble(-0.264544, -0.0119817),
            w1Low = cms.vdouble(0.239183, -0.00807308)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.0187558, -0.00974396),
            w1Up = cms.vdouble(-0.0387561, -0.00125678),
            w0Low = cms.vdouble(-0.278701, -0.0123996),
            w1Low = cms.vdouble(0.327602, -0.0059854)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0718516, -0.0106175),
            w1Up = cms.vdouble(0.0518574, -0.00738235),
            w0Low = cms.vdouble(-0.348819, -0.0142049),
            w1Low = cms.vdouble(0.328818, -0.00361751)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0805835, -0.0107719),
            w1Up = cms.vdouble(0.0605829, -0.0051451),
            w0Low = cms.vdouble(-0.458802, -0.0149081),
            w1Low = cms.vdouble(0.438874, -0.00509102)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.015616, -0.00118436),
            w1Up = cms.vdouble(-0.0555665, 0.000130688),
            w0Low = cms.vdouble(-0.201599, -0.0122628),
            w1Low = cms.vdouble(0.440765, -0.00182968)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.0795331, -0.00886947),
            w1Up = cms.vdouble(-0.0701426, -0.00900796),
            w0Low = cms.vdouble(-0.250664, -0.00509319),
            w1Low = cms.vdouble(0.258523, 0.00409318)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.288495, -0.00637448),
            w1Up = cms.vdouble(-0.364588, -0.00563395),
            w0Low = cms.vdouble(-0.0453384, -0.0104732),
            w1Low = cms.vdouble(0.0213177, 0.00341946)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.176263, -0.00737006),
            w1Up = cms.vdouble(-0.196314, -0.0126305),
            w0Low = cms.vdouble(-0.24465, -0.0146882),
            w1Low = cms.vdouble(0.171963, -0.00643569)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.200978, -0.00687581),
            w1Up = cms.vdouble(-0.220972, -0.013124),
            w0Low = cms.vdouble(-0.314477, -0.0122891),
            w1Low = cms.vdouble(0.144614, -0.00471034)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(0.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429,-0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0099678, -0.00999779),
            w1Up = cms.vdouble(-0.0100322, -0.00900221),
            w0Low = cms.vdouble(-0.0203683, -0.0107285),
            w1Low = cms.vdouble(0.000368696, -0.00428797)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.06495, -0.0143591),
            w1Up = cms.vdouble(0.0601983, -0.00293327),
            w0Low = cms.vdouble(-0.0638032, -0.0142664),
            w1Low = cms.vdouble(0.0438029, -0.00412073)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.062143, -0.0146226),
            w1Up = cms.vdouble(0.0421399, -0.00537746),
            w0Low = cms.vdouble(-0.0312307, -0.0044175),
            w1Low = cms.vdouble(0.0401687, 0.00319768)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0464141, -0.0135373),
            w1Up = cms.vdouble(0.0207001, -0.00446349),
            w0Low = cms.vdouble(-0.0496976, -0.0138553),
            w1Low = cms.vdouble(0.0296941, -0.00414534)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0113465, -0.0025782),
            w1Up = cms.vdouble(0.0190533, 0.00153546),
            w0Low = cms.vdouble(-0.0250704, -0.00445547),
            w1Low = cms.vdouble(0.0369043, 0.00332655)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00475482, -0.000826044),
            w1Up = cms.vdouble(0.00310358, 9.00178-05),
            w0Low = cms.vdouble(-0.0330374, -0.0125854),
            w1Low = cms.vdouble(0.0208555, -0.00353981)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.000587405, -0.00778347),
            w1Up = cms.vdouble(-0.0286249, -0.0123259),
            w0Low = cms.vdouble(-0.0304373, -0.0124313),
            w1Low = cms.vdouble(0.0104358, -0.00456935)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00917579, -0.00673796),
            w1Up = cms.vdouble(-0.0360177, -0.0112618),
            w0Low = cms.vdouble(-0.0183648, -0.00396966),
            w1Low = cms.vdouble(0.0290483, 0.00277955)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.0266681, 0.00224996),
            w1Up = cms.vdouble(-0.016518, -0.00360951),
            w0Low = cms.vdouble(0.000622076, -0.00860274),
            w1Low = cms.vdouble(-0.0206216, 0.0026033)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(0.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00275589, -0.000416252),
            w1Up = cms.vdouble(0.00511922, -0.000641199),
            w0Low = cms.vdouble(-0.0122405, -0.00321189),
            w1Low = cms.vdouble(0.0201, 0.00221189)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.159652, -0.0107142),
            w1Up = cms.vdouble(-0.00663801, -0.00926284),
            w0Low = cms.vdouble(-0.0348881, -0.0153363),
            w1Low = cms.vdouble(0.017326, -0.0021408)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.02465, -0.00665416),
            w1Up = cms.vdouble(0.0325779, 0.00556419),
            w0Low = cms.vdouble(-0.0203151, -0.00544499),
            w1Low = cms.vdouble(0.0283403, 0.00458016)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0768058, -0.0113343),
            w1Up = cms.vdouble(-0.00560368, -0.00918081),
            w0Low = cms.vdouble(-0.0314732, -0.0146387),
            w1Low = cms.vdouble(0.0114405, -0.0051577)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0421559, -0.0103338),
            w1Up = cms.vdouble(-0.0089238, -0.00969239),
            w0Low = cms.vdouble(-0.0156738, -0.0116092),
            w1Low = cms.vdouble(-0.00302215, 0.000609279)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0102848, -0.0100827),
            w1Up = cms.vdouble(-0.00971479, -0.00891351),
            w0Low = cms.vdouble(-0.0232224, -0.0131033),
            w1Low = cms.vdouble(0.00421815, -0.0059919)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00736665, -0.00915309),
            w1Up = cms.vdouble(-0.0126323, -0.0129284),
            w0Low = cms.vdouble(-0.0100599, -0.0100192),
            w1Low = cms.vdouble(-0.00994014, 0.00401972)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00411853, -0.00411128),
            w1Up = cms.vdouble(-0.0227719, -0.00588691),
            w0Low = cms.vdouble(-0.0194847, -0.0132155),
            w1Low = cms.vdouble(-0.00947776, -0.00271955)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.0117318, 0.00265123),
            w1Up = cms.vdouble(-0.00399454, -0.00362366),
            w0Low = cms.vdouble(-0.0071709, -0.00243773),
            w1Low = cms.vdouble(0.0148231, 0.00141334)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00144727, -0.00631455),
            w1Up = cms.vdouble(-0.0350447, -0.0115964),
            w0Low = cms.vdouble(-0.0175174, -0.0127914),
            w1Low = cms.vdouble(-0.00248271, -0.00720808)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(0.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0309894, -0.019105),
            w1Up = cms.vdouble(0.00946211, -0.00159988),
            w0Low = cms.vdouble(-0.0170101, -0.00655528),
            w1Low = cms.vdouble(0.0242961, 0.00532055)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0105939, -0.00389898),
            w1Up = cms.vdouble(0.0201303, 0.00753434),
            w0Low = cms.vdouble(-0.0216746, -0.0152646),
            w1Low = cms.vdouble(0.000270993, 0.001241)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0078299, -0.00295277),
            w1Up = cms.vdouble(0.0155669, 0.00198999),
            w0Low = cms.vdouble(-0.0212003, -0.0155353),
            w1Low = cms.vdouble(0.00312757, -0.00351509)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00972848, -0.00985747),
            w1Up = cms.vdouble(-0.0102715, -0.000144856),
            w0Low = cms.vdouble(-0.0185711, -0.0145963),
            w1Low = cms.vdouble(-0.0013883, -0.00533342)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00204754, 0.00130331),
            w1Up = cms.vdouble(0.00774106, -0.000797305),
            w0Low = cms.vdouble(-0.0115078, -0.0106757),
            w1Low = cms.vdouble(-0.00763415, 0.00434953)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00417311, 0.00282262),
            w1Up = cms.vdouble(0.00560022, -0.00445311),
            w0Low = cms.vdouble(-0.0133823, -0.012041),
            w1Low = cms.vdouble(-0.00662405, -0.00136526)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00303689, -0.00555942),
            w1Up = cms.vdouble(-0.0169639, -0.00844213),
            w0Low = cms.vdouble(-0.00365493, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.000999995)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00628935, 0.00478926),
            w1Up = cms.vdouble(0.0015273, -0.00519179),
            w0Low = cms.vdouble(-0.0150102, -0.0133511),
            w1Low = cms.vdouble(-0.00498967, -0.00564879)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00434825, -0.00605298),
            w1Up = cms.vdouble(-0.0155785, -0.0128982),
            w0Low = cms.vdouble(-0.00244256, 0.000285661),
            w1Low = cms.vdouble(0.0114891, -0.00046182)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(0.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0146397, -0.00605089),
            w1Up = cms.vdouble(0.0200071, 0.00536041),
            w0Low = cms.vdouble(-0.0245375, -0.0191676),
            w1Low = cms.vdouble(0.004538, 0.000167113)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.016679, -0.0166297),
            w1Up = cms.vdouble(-0.000595417, -0.00237509),
            w0Low = cms.vdouble(-0.0132095, -0.00753693),
            w1Low = cms.vdouble(0.0203061, 0.00653704)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0153183, -0.0118562),
            w1Up = cms.vdouble(-0.00667009, -0.00143257),
            w0Low = cms.vdouble(-0.00879608, -0.00409118),
            w1Low = cms.vdouble(0.0166422, 0.00422281)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00148084, 0.00153045),
            w1Up = cms.vdouble(0.0104362, -0.000409696),
            w0Low = cms.vdouble(-0.0163657, -0.0153129),
            w1Low = cms.vdouble(-0.00321395, -0.00433932)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00875332, -0.00888821),
            w1Up = cms.vdouble(-0.0112508, -0.00611544),
            w0Low = cms.vdouble(-0.0153905, -0.0148011),
            w1Low = cms.vdouble(-0.00459607, -0.00520545)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00090881, 0.00417119),
            w1Up = cms.vdouble(0.00343727, -0.00517156),
            w0Low = cms.vdouble(-0.00539317, 0.00179848),
            w1Low = cms.vdouble(0.0128595, 0.000640759)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00677793, -0.0042296),
            w1Up = cms.vdouble(-0.0122412, -0.0082216),
            w0Low = cms.vdouble(-0.00449344, 0.000693919),
            w1Low = cms.vdouble(0.0108706, 0.000384456)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00149129, 0.00550409),
            w1Up = cms.vdouble(0.00384094, -0.0117169),
            w0Low = cms.vdouble(-0.0148652, -0.0107762),
            w1Low = cms.vdouble(-0.00922824, -0.0041843)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00271772, 0.00891708),
            w1Up = cms.vdouble(0.00426737, -0.00691003),
            w0Low = cms.vdouble(-0.0117726, -0.011936),
            w1Low = cms.vdouble(-0.00822764, -0.00806427)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(0.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0107405, -0.00702766),
            w1Up = cms.vdouble(0.0192701, 0.00599729),
            w0Low = cms.vdouble(-0.0210618, -0.0197011),
            w1Low = cms.vdouble(0.0011167, -0.000212889)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00808364, -0.00394184),
            w1Up = cms.vdouble(0.0159945, 0.00382172),
            w0Low = cms.vdouble(-0.00944302, -0.00565521),
            w1Low = cms.vdouble(0.0181547, 0.00466661)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0147516, -0.0151025),
            w1Up = cms.vdouble(-0.0052862, -0.00465631),
            w0Low = cms.vdouble(-0.00647021, -0.00285123),
            w1Low = cms.vdouble(0.0144018, 0.00200023)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0104565, -0.0221006),
            w1Up = cms.vdouble(-0.00976258, -0.00847093),
            w0Low = cms.vdouble(-0.0150041, -0.0148114),
            w1Low = cms.vdouble(-0.00532524, -0.00350829)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0103182, -0.0104603),
            w1Up = cms.vdouble(-0.00963624, -0.00444847),
            w0Low = cms.vdouble(-0.00434411, 0.000912861),
            w1Low = cms.vdouble(0.012698, 0.000354741)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0103316, -0.0104401),
            w1Up = cms.vdouble(-0.00947763, -0.00831151),
            w0Low = cms.vdouble(-0.0128453, -0.0137421),
            w1Low = cms.vdouble(-0.00715479, -0.00325776)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00144847, 0.00593938),
            w1Up = cms.vdouble(0.00777765, -0.00664315),
            w0Low = cms.vdouble(-0.00257594, 0.00235184),
            w1Low = cms.vdouble(0.0114483, -0.00062785)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.000952265, 0.00804162),
            w1Up = cms.vdouble(0.00730788, -0.0111115),
            w0Low = cms.vdouble(-0.000314776, -0.000192345),
            w1Low = cms.vdouble(0.0123094, -0.00184148)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00259784, 0.0102396),
            w1Up = cms.vdouble(0.00539583, -0.00703802),
            w0Low = cms.vdouble(-0.0127815, -0.0142242),
            w1Low = cms.vdouble(-0.00828354, -0.00739318)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00208404, 0.0137645),
            w1Up = cms.vdouble(0.00512926, -0.00615535),
            w0Low = cms.vdouble(-0.0143414, -0.0110661),
            w1Low = cms.vdouble(-0.0093385, -0.00861006)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(1.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00981215, -0.00676206),
            w1Up = cms.vdouble(0.018251, 0.00560551),
            w0Low = cms.vdouble(-0.0113791, -0.00875908),
            w1Low = cms.vdouble(0.0187593, 0.0075375)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0166951, -0.0178789),
            w1Up = cms.vdouble(-0.00352396, 0.00326515),
            w0Low = cms.vdouble(-0.0205273, -0.0233098),
            w1Low = cms.vdouble(-0.000357508, 0.00210654)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0141875, -0.015795),
            w1Up = cms.vdouble(-0.00598675, -0.00375859),
            w0Low = cms.vdouble(-0.0170269, -0.0195912),
            w1Low = cms.vdouble(-0.00336375, -0.00101741)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0107691, -0.0146503),
            w1Up = cms.vdouble(-0.00845498, -0.00425711),
            w0Low = cms.vdouble(-0.00571911, -0.00197537),
            w1Low = cms.vdouble(0.0135764, -0.00274194)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0105371, -0.0112451),
            w1Up = cms.vdouble(-0.00886341, -0.00790671),
            w0Low = cms.vdouble(-0.00396673, 0.00231649),
            w1Low = cms.vdouble(0.0108749, 0.000159805)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00884569, -0.00806587),
            w1Up = cms.vdouble(-0.0107507, -0.00193018),
            w0Low = cms.vdouble(-0.00354571, 0.00164434),
            w1Low = cms.vdouble(0.0106025, -0.00264395)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.000521926, 0.00670527),
            w1Up = cms.vdouble(0.00788949, -0.00918363),
            w0Low = cms.vdouble(-0.00256545, 0.00137239),
            w1Low = cms.vdouble(0.0106876, -0.00457911)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00921301, -0.00853663),
            w1Up = cms.vdouble(-0.0110079, -0.0122919),
            w0Low = cms.vdouble(-0.0135151, -0.0139868),
            w1Low = cms.vdouble(-0.00977354, -0.00357986)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00380383, 0.0111064),
            w1Up = cms.vdouble(0.0074252, -0.010619),
            w0Low = cms.vdouble(-0.00330164, 0.0132875),
            w1Low = cms.vdouble(0.0120649, -0.00385437)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00780011, -0.00542804),
            w1Up = cms.vdouble(-0.0121732, -0.0135518),
            w0Low = cms.vdouble(-0.00254516, 0.00621414),
            w1Low = cms.vdouble(0.0078796, -0.00366267)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00243535, 0.0151118),
            w1Up = cms.vdouble(0.00561083, -0.00653158),
            w0Low = cms.vdouble(-0.0104004, -0.0114967),
            w1Low = cms.vdouble(-0.00928473, -0.0031029)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(1.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0119549, -0.0122793),
            w1Up = cms.vdouble(-0.0080779, 1.70947-05),
            w0Low = cms.vdouble(-0.0117829, -0.0101207),
            w1Low = cms.vdouble(0.0196342, 0.00911389)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0233533, -0.0299742),
            w1Up = cms.vdouble(0.00301185, 0.0139831),
            w0Low = cms.vdouble(-0.00979742, -0.00891987),
            w1Low = cms.vdouble(0.0195228, 0.0210743)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0177314, -0.0227774),
            w1Up = cms.vdouble(-0.000916665, 0.0148593),
            w0Low = cms.vdouble(-0.00723788, -0.00502347),
            w1Low = cms.vdouble(0.0155126, 0.025988)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00669711, -0.00376907),
            w1Up = cms.vdouble(0.01472, 0.00368092),
            w0Low = cms.vdouble(-0.00804219, -0.0061207),
            w1Low = cms.vdouble(0.0158989, 0.00404979)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00438699, 0.000512196),
            w1Up = cms.vdouble(0.0122913, -0.00197699),
            w0Low = cms.vdouble(-0.00567834, -0.0019316),
            w1Low = cms.vdouble(0.0135209, 0.000907146)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00269333, 0.00198052),
            w1Up = cms.vdouble(0.0116042, -0.00295123),
            w0Low = cms.vdouble(-0.0144403, -0.0189826),
            w1Low = cms.vdouble(-0.0059203, -0.00181032)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00254935, 0.00913051),
            w1Up = cms.vdouble(0.0106433, -0.00212626),
            w0Low = cms.vdouble(-0.0021628, 0.00214933),
            w1Low = cms.vdouble(0.0114217, -0.00145455)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0101148, -0.0102508),
            w1Up = cms.vdouble(-0.00985796, -0.00568493),
            w0Low = cms.vdouble(-0.0127606, -0.0151101),
            w1Low = cms.vdouble(-0.00771878, -0.00564813)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00813286, -0.00887433),
            w1Up = cms.vdouble(-0.0107246, -0.00968506),
            w0Low = cms.vdouble(-0.00271925, 0.00515541),
            w1Low = cms.vdouble(0.0108131, -0.00715165)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.000391724, 0.0150327),
            w1Up = cms.vdouble(0.00808143, -0.0133848),
            w0Low = cms.vdouble(-0.00232308, 0.00620545),
            w1Low = cms.vdouble(0.0105167, -0.00715069)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00100498, 0.0147365),
            w1Up = cms.vdouble(0.00720589, -0.0162301),
            w0Low = cms.vdouble(-0.0114388, -0.0136023),
            w1Low = cms.vdouble(-0.00725066, -0.00638851)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(1.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00467328, 0.000155302),
            w1Up = cms.vdouble(0.0127576, -0.0645623),
            w0Low = cms.vdouble(-0.00417894, 0.00207067),
            w1Low = cms.vdouble(0.0120351, -0.0243557)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0141871, -0.014322),
            w1Up = cms.vdouble(-0.00582644, -0.00552002),
            w0Low = cms.vdouble(-0.0319325, -0.0470354),
            w1Low = cms.vdouble(-0.00890179, -0.0081486)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0120631, -0.0137376),
            w1Up = cms.vdouble(-0.00775106, -0.000632721),
            w0Low = cms.vdouble(-0.0195812, -0.0311028),
            w1Low = cms.vdouble(-0.00025393, 0.0130996)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0109648, -0.0124764),
            w1Up = cms.vdouble(-0.00893558, -0.00689097),
            w0Low = cms.vdouble(-0.00837879, -0.00703606),
            w1Low = cms.vdouble(0.0158521, 0.0045536)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0102624, -0.0108302),
            w1Up = cms.vdouble(-0.00968424, -0.00925139),
            w0Low = cms.vdouble(-0.00551179, -0.0016656),
            w1Low = cms.vdouble(0.0133731, 0.000513272)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.010569, -0.0115906),
            w1Up = cms.vdouble(-0.00943245, -0.00790778),
            w0Low = cms.vdouble(-0.0141244, -0.0190176),
            w1Low = cms.vdouble(-0.00590191, -0.000887662)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0094844, -0.00874637),
            w1Up = cms.vdouble(-0.0105473, -0.00180626),
            w0Low = cms.vdouble(-0.00371535, -5.20843-05),
            w1Low = cms.vdouble(0.0099992, -0.000311885)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.000252059, 0.00928316),
            w1Up = cms.vdouble(0.00858432, -0.00987321),
            w0Low = cms.vdouble(-0.0031014, 0.0049917),
            w1Low = cms.vdouble(0.0106507, 1.33126-05)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.000415236, 0.0114064),
            w1Up = cms.vdouble(0.0079842, -0.0127877),
            w0Low = cms.vdouble(-0.00381221, 0.00625311),
            w1Low = cms.vdouble(0.010292, -0.00771069)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.000903548, 0.0144743),
            w1Up = cms.vdouble(0.00746167, -0.0175398),
            w0Low = cms.vdouble(0.000615912, 0.00356772),
            w1Low = cms.vdouble(0.0107108, -0.00438631)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.000472288, 0.00966081),
            w1Up = cms.vdouble(0.00849141, -0.0265544),
            w0Low = cms.vdouble(0.000338268, 0.00412686),
            w1Low = cms.vdouble(0.0101243, -0.00522594)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(1.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00106129, 0.017786),
            w1Up = cms.vdouble(0.00679553, -0.0187856),
            w0Low = cms.vdouble(-0.00310141, 0.00574806),
            w1Low = cms.vdouble(0.0109179, -0.0178981)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00440644, 0.00496092),
            w1Up = cms.vdouble(0.00749941, 0.000578433),
            w0Low = cms.vdouble(-0.00592228, -0.00238797),
            w1Low = cms.vdouble(0.0137787, 0.0013731)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0109966, -0.0116698),
            w1Up = cms.vdouble(-0.00911351, -0.00181474),
            w0Low = cms.vdouble(-0.0186469, -0.0270812),
            w1Low = cms.vdouble(-0.00134313, 0.00715923)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0076422, -0.00977104),
            w1Up = cms.vdouble(-0.0126291, 0.000837955),
            w0Low = cms.vdouble(-0.0181204, -0.0272754),
            w1Low = cms.vdouble(-0.0021117, 0.00681343)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00999189, -0.00998596),
            w1Up = cms.vdouble(-0.0100076, -0.00801742),
            w0Low = cms.vdouble(-0.00556468, -0.00222402),
            w1Low = cms.vdouble(0.0135891, 0.000991235)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0106308, -0.012271),
            w1Up = cms.vdouble(-0.00924478, -0.00047817),
            w0Low = cms.vdouble(-0.00559993, -0.001728),
            w1Low = cms.vdouble(0.0134571, 0.000727407)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00161442, 0.00373603),
            w1Up = cms.vdouble(0.00948149, -0.00658234),
            w0Low = cms.vdouble(-0.0052281, -0.00108042),
            w1Low = cms.vdouble(0.0131358, -0.000380837)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.000509061, 0.00467741),
            w1Up = cms.vdouble(0.00835078, -0.0104224),
            w0Low = cms.vdouble(-0.0135657, -0.0186656),
            w1Low = cms.vdouble(-0.00648478, -0.000628699)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00124663, 0.00875456),
            w1Up = cms.vdouble(0.0097028, -0.0121833),
            w0Low = cms.vdouble(-0.00464921, 0.0109702),
            w1Low = cms.vdouble(0.0125152, -0.001892)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00204418, 0.00768681),
            w1Up = cms.vdouble(0.00579388, -0.00166465),
            w0Low = cms.vdouble(-0.00164021, 0.00229781),
            w1Low = cms.vdouble(0.0106349, 0.000328627)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.00163766, 0.0210828),
            w1Up = cms.vdouble(0.00766689, -0.00883815),
            w0Low = cms.vdouble(-0.00106073, 0.00334185),
            w1Low = cms.vdouble(0.0118373, -0.0129547)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(1.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0109567, -0.0119789),
            w1Up = cms.vdouble(-0.00900765, -0.00805727),
            w0Low = cms.vdouble(-0.00666924, -0.00412925),
            w1Low = cms.vdouble(0.0146376, 0.00312243)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0106007, -0.0119047),
            w1Up = cms.vdouble(-0.0081759, -0.00522692),
            w0Low = cms.vdouble(-0.00709371, -0.00468485),
            w1Low = cms.vdouble(0.0150732, 0.0234139)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0111713, -0.0137497),
            w1Up = cms.vdouble(-0.00837203, -0.00683784),
            w0Low = cms.vdouble(-0.0078205, -0.00706718),
            w1Low = cms.vdouble(0.0156508, 0.00707047)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0118529, -0.0145074),
            w1Up = cms.vdouble(-0.00815945, 0.00247663),
            w0Low = cms.vdouble(-0.0171571, -0.0274055),
            w1Low = cms.vdouble(-0.00284131, 0.00841025)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00455769, 0.00059374),
            w1Up = cms.vdouble(0.0127329, -0.000297407),
            w0Low = cms.vdouble(-0.00721206, -0.00716293),
            w1Low = cms.vdouble(0.0162813, 0.00188149)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.013914, -0.022346),
            w1Up = cms.vdouble(-0.00500323, 0.0015054),
            w0Low = cms.vdouble(-0.00650964, -0.00451567),
            w1Low = cms.vdouble(0.014685, 0.00192503)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00486329, 0.000196462),
            w1Up = cms.vdouble(0.0128017, -0.0114336),
            w0Low = cms.vdouble(-0.017091, -0.0228374),
            w1Low = cms.vdouble(-0.00542851, 0.00550445)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00148083, 0.0103629),
            w1Up = cms.vdouble(0.0108625, -0.0113984),
            w0Low = cms.vdouble(-0.00363611, 7.71175-05),
            w1Low = cms.vdouble(0.0130948, -0.00110403)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(0.000591976, 0.0171286),
            w1Up = cms.vdouble(0.00727446, -0.0180998),
            w0Low = cms.vdouble(-0.0116986, -0.0152393),
            w1Low = cms.vdouble(-0.0083496, -0.00499487)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(2.),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00382658, 0.000527324),
            w1Up = cms.vdouble(0.0113824, 0.000773746),
            w0Low = cms.vdouble(-0.00742432, -0.00611724),
            w1Low = cms.vdouble(0.0153796, 0.00363947)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0115931, -0.0143562),
            w1Up = cms.vdouble(-0.00821427, -0.0063927),
            w0Low = cms.vdouble(-0.00796472, -0.00706351),
            w1Low = cms.vdouble(0.0159966, 0.0171043)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0116423, -0.0137977),
            w1Up = cms.vdouble(-0.00690602, -0.000858004),
            w0Low = cms.vdouble(-0.0191502, -0.0310441),
            w1Low = cms.vdouble(-0.000986918, 0.0107482)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00374852, 0.00277302),
            w1Up = cms.vdouble(0.0116486, -0.00371779),
            w0Low = cms.vdouble(-0.00989501, -0.0121418),
            w1Low = cms.vdouble(0.017175, 0.0111401)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00528787, -0.000806429),
            w1Up = cms.vdouble(0.0131637, -0.00305324),
            w0Low = cms.vdouble(-0.00900516, -0.00819726),
            w1Low = cms.vdouble(0.017198, 0.0127234)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(2.2),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00671979, -0.00380751),
            w1Up = cms.vdouble(-0.00861281, -0.000223681),
            w0Low = cms.vdouble(-0.00962149, -0.0097846),
            w1Low = cms.vdouble(0.0176851, 0.00282043)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00586597, -0.00224627),
            w1Up = cms.vdouble(0.0136995, 0.00113298),
            w0Low = cms.vdouble(-0.0119883, -0.0132702),
            w1Low = cms.vdouble(0.0192354, 0.0127335)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00921337, -0.00972774),
            w1Up = cms.vdouble(0.0174732, 0.00374424),
            w0Low = cms.vdouble(-0.0111791, -0.0141913),
            w1Low = cms.vdouble(0.0196535, 0.0234483)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00705733, -0.00552882),
            w1Up = cms.vdouble(0.0153107, 0.00479939),
            w0Low = cms.vdouble(-0.0068647, -0.0044674),
            w1Low = cms.vdouble(0.014891, 0.030301)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(2.4),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0125923, -0.0112565),
            w1Up = cms.vdouble(0.0205578, 0.00362254),
            w0Low = cms.vdouble(-0.0107444, -0.0174383),
            w1Low = cms.vdouble(0.018352, 0.00836721)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0162762, -0.0245287),
            w1Up = cms.vdouble(0.0234757, 0.0184419),
            w0Low = cms.vdouble(-0.0179058, -0.023449),
            w1Low = cms.vdouble(0.0251013, 0.0111108)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0263748, -0.0262169),
            w1Up = cms.vdouble(0.0343395, 0.0351252),
            w0Low = cms.vdouble(-0.0196697, -0.0263439),
            w1Low = cms.vdouble(0.027347, 0.032635)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0275095, -0.0274427),
            w1Up = cms.vdouble(0.0399894, 0.00499809),
            w0Low = cms.vdouble(-0.0173567, -0.0228128),
            w1Low = cms.vdouble(0.025214, 0.0121381)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(2.6),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-1.),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.8),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.6),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.4),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(-0.2),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.2),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.4),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.6),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(0.8),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0158867, -0.0108003),
            w1Up = cms.vdouble(0.0239253, 0.00644046),
            w0Low = cms.vdouble(-0.0179981, -0.0144231),
            w1Low = cms.vdouble(0.0253099, 0.0175108)
        ),
        cms.PSet(
            log10EMin = cms.double(1.),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.025836, -0.021872),
            w1Up = cms.vdouble(0.0328125, 0.00506957),
            w0Low = cms.vdouble(-0.0204996, -0.0180844),
            w1Low = cms.vdouble(0.0278642, 0.00846112)
        ),
        cms.PSet(
            log10EMin = cms.double(1.2),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.0211318, -0.0185802),
            w1Up = cms.vdouble(0.0290236, 0.0244123),
            w0Low = cms.vdouble(-0.0199571, -0.0156955),
            w1Low = cms.vdouble(0.0274815, 0.0163092)
        ),
        cms.PSet(
            log10EMin = cms.double(1.4),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.6),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        ),
        cms.PSet(
            log10EMin = cms.double(1.8),
            etaMin = cms.double(2.8),
            pUp = cms.vdouble(-0.107537, 0.590969, -0.076494),
            pLow = cms.vdouble(-0.0268843, 0.147742, -0.0191235),
            w0Up = cms.vdouble(-0.00571429, -0.002),
            w1Up = cms.vdouble(0.0135714, 0.001),
            w0Low = cms.vdouble(-0.00571429, -0.002),
            w1Low = cms.vdouble(0.0135714, 0.001)
        )
    )
)

