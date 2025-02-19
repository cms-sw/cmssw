import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.stringResolutionProvider_cfi import *

## <--- these resolutions do not exist yet --->
## electronResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
##                                        resolutions     = ['et * (sqrt(5.6*5.6/(et*et) + 1.25/et + 0.033))', # add sigma(Et) not sigma(Et)/Et here
##                                                           '0.03  + 1.0/et',                                 # add sigma(eta) here
##                                                           '0.015 + 1.5/et'                                  # add sigma(phi) here
##                                                           ],
##                                        constraints     =  cms.vdouble(0)                                    # add constraints here
##                                        )

muonResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.100'),
    et  = cms.string('et * (0.00465 + 0.0002471 * et)'),
    eta = cms.string('sqrt(0.0004331^2 + (0.001071/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(6.21e-05^2 + (0/sqrt(et))^2 + (0.004634/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.100<=abs(eta) && abs(eta)<0.200'),
    et  = cms.string('et * (0.005072 + 0.0002368 * et)'),
    eta = cms.string('sqrt(0.0003896^2 + (0.000858/sqrt(et))^2 + (0.00201/et)^2)'),
    phi = cms.string('sqrt(5.36e-05^2 + (0/sqrt(et))^2 + (0.004865/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.200<=abs(eta) && abs(eta)<0.300'),
    et  = cms.string('et * (0.005875 + 0.0002207 * et)'),
    eta = cms.string('sqrt(0.0003387^2 + (0.000904/sqrt(et))^2 + (0.00142/et)^2)'),
    phi = cms.string('sqrt(5.16e-05^2 + (0/sqrt(et))^2 + (0.004923/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.300<=abs(eta) && abs(eta)<0.400'),
    et  = cms.string('et * (0.006974 + 0.0002021 * et)'),
    eta = cms.string('sqrt(0.0003164^2 + (0.000704/sqrt(et))^2 + (0.00169/et)^2)'),
    phi = cms.string('sqrt(5.21e-05^2 + (0/sqrt(et))^2 + (0.005102/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.400<=abs(eta) && abs(eta)<0.500'),
    et  = cms.string('et * (0.007159 + 0.0002023 * et)'),
    eta = cms.string('sqrt(0.0002926^2 + (0.000722/sqrt(et))^2 + (0.00154/et)^2)'),
    phi = cms.string('sqrt(5.3e-05^2 + (0/sqrt(et))^2 + (0.005151/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.500<=abs(eta) && abs(eta)<0.600'),
    et  = cms.string('et * (0.007502 + 0.000193 * et)'),
    eta = cms.string('sqrt(0.0002897^2 + (0.000754/sqrt(et))^2 + (0.00159/et)^2)'),
    phi = cms.string('sqrt(5.32e-05^2 + (0/sqrt(et))^2 + (0.005276/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.600<=abs(eta) && abs(eta)<0.700'),
    et  = cms.string('et * (0.007842 + 0.0001886 * et)'),
    eta = cms.string('sqrt(0.0003089^2 + (0.000684/sqrt(et))^2 + (0.00189/et)^2)'),
    phi = cms.string('sqrt(5.31e-05^2 + (0/sqrt(et))^2 + (0.00538/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.700<=abs(eta) && abs(eta)<0.800'),
    et  = cms.string('et * (0.008325 + 0.0001833 * et)'),
    eta = cms.string('sqrt(0.00029^2 + (0.000868/sqrt(et))^2 + (0.00181/et)^2)'),
    phi = cms.string('sqrt(5.54e-05^2 + (0/sqrt(et))^2 + (0.005242/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.800<=abs(eta) && abs(eta)<0.900'),
    et  = cms.string('et * (0.00925 + 0.0001917 * et)'),
    eta = cms.string('sqrt(0.0002935^2 + (0.000783/sqrt(et))^2 + (0.00204/et)^2)'),
    phi = cms.string('sqrt(6.05e-05^2 + (0/sqrt(et))^2 + (0.00561/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.900<=abs(eta) && abs(eta)<1.000'),
    et  = cms.string('et * (0.01095 + 0.000192 * et)'),
    eta = cms.string('sqrt(0.0002772^2 + (0.000916/sqrt(et))^2 + (0.00149/et)^2)'),
    phi = cms.string('sqrt(7.7e-05^2 + (0/sqrt(et))^2 + (0.005576/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.000<=abs(eta) && abs(eta)<1.100'),
    et  = cms.string('et * (0.01267 + 0.0001638 * et)'),
    eta = cms.string('sqrt(0.0002908^2 + (0.000919/sqrt(et))^2 + (0.0018/et)^2)'),
    phi = cms.string('sqrt(7.53e-05^2 + (0/sqrt(et))^2 + (0.005775/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.100<=abs(eta) && abs(eta)<1.200'),
    et  = cms.string('et * (0.01374 + 0.0001666 * et)'),
    eta = cms.string('sqrt(0.0002931^2 + (0.000943/sqrt(et))^2 + (0.00201/et)^2)'),
    phi = cms.string('sqrt(8.18e-05^2 + (0/sqrt(et))^2 + (0.00606/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.200<=abs(eta) && abs(eta)<1.300'),
    et  = cms.string('et * (0.01492 + 0.0001584 * et)'),
    eta = cms.string('sqrt(0.0002936^2 + (0.000794/sqrt(et))^2 + (0.00214/et)^2)'),
    phi = cms.string('sqrt(7.43e-05^2 + (0.000429/sqrt(et))^2 + (0.006143/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.300<=abs(eta) && abs(eta)<1.400'),
    et  = cms.string('et * (0.01535 + 0.0001721 * et)'),
    eta = cms.string('sqrt(0.0002927^2 + (0.000856/sqrt(et))^2 + (0.0023/et)^2)'),
    phi = cms.string('sqrt(5.6e-05^2 + (0.000724/sqrt(et))^2 + (0.005829/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.400<=abs(eta) && abs(eta)<1.500'),
    et  = cms.string('et * (0.01477 + 0.0001847 * et)'),
    eta = cms.string('sqrt(0.0003012^2 + (0.000872/sqrt(et))^2 + (0.00229/et)^2)'),
    phi = cms.string('sqrt(4.9e-05^2 + (0.000815/sqrt(et))^2 + (0.005459/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.500<=abs(eta) && abs(eta)<1.600'),
    et  = cms.string('et * (0.01353 + 0.0002351 * et)'),
    eta = cms.string('sqrt(0.0003027^2 + (0.00077/sqrt(et))^2 + (0.00258/et)^2)'),
    phi = cms.string('sqrt(7.63e-05^2 + (0.000745/sqrt(et))^2 + (0.005747/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.600<=abs(eta) && abs(eta)<1.700'),
    et  = cms.string('et * (0.01301 + 0.0002884 * et)'),
    eta = cms.string('sqrt(0.0003074^2 + (0.000725/sqrt(et))^2 + (0.00271/et)^2)'),
    phi = cms.string('sqrt(7.5e-05^2 + (0.000967/sqrt(et))^2 + (0.00546/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.700<=abs(eta) && abs(eta)<1.800'),
    et  = cms.string('et * (0.01295 + 0.000388 * et)'),
    eta = cms.string('sqrt(0.0003168^2 + (0.00067/sqrt(et))^2 + (0.00295/et)^2)'),
    phi = cms.string('sqrt(0.000109^2 + (0.000978/sqrt(et))^2 + (0.00572/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.800<=abs(eta) && abs(eta)<1.900'),
    et  = cms.string('et * (0.01382 + 0.000506 * et)'),
    eta = cms.string('sqrt(0.000344^2 + (0.00063/sqrt(et))^2 + (0.00303/et)^2)'),
    phi = cms.string('sqrt(0.000134^2 + (0.001074/sqrt(et))^2 + (0.00561/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.900<=abs(eta) && abs(eta)<2.000'),
    et  = cms.string('et * (0.01519 + 0.000565 * et)'),
    eta = cms.string('sqrt(0.000337^2 + (0.00087/sqrt(et))^2 + (0.00322/et)^2)'),
    phi = cms.string('sqrt(0.000186^2 + (0.00084/sqrt(et))^2 + (0.0061/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.000<=abs(eta) && abs(eta)<2.100'),
    et  = cms.string('et * (0.01712 + 0.000755 * et)'),
    eta = cms.string('sqrt(0.00036^2 + (0.00065/sqrt(et))^2 + (0.00393/et)^2)'),
    phi = cms.string('sqrt(0.000216^2 + (0.00124/sqrt(et))^2 + (0.00572/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.100<=abs(eta) && abs(eta)<2.200'),
    et  = cms.string('et * (0.01979 + 0.00085 * et)'),
    eta = cms.string('sqrt(0.000372^2 + (0.00096/sqrt(et))^2 + (0.0037/et)^2)'),
    phi = cms.string('sqrt(0.00031^2 + (0.00072/sqrt(et))^2 + (0.0062/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.200<=abs(eta) && abs(eta)<2.300'),
    et  = cms.string('et * (0.02143 + 0.00109 * et)'),
    eta = cms.string('sqrt(0.000432^2 + (0.00063/sqrt(et))^2 + (0.00447/et)^2)'),
    phi = cms.string('sqrt(0.000333^2 + (0.00146/sqrt(et))^2 + (0.00566/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.300<=abs(eta) && abs(eta)<2.400'),
    et  = cms.string('et * (0.02144 + 0.001458 * et)'),
    eta = cms.string('sqrt(0.000417^2 + (0.00149/sqrt(et))^2 + (0.00416/et)^2)'),
    phi = cms.string('sqrt(0.000365^2 + (0.00172/sqrt(et))^2 + (0.00628/et)^2)'),
    )
    ),
                                        constraints = cms.vdouble(0)
                                        )

## <--- these resolutions do not exist yet --->
## tauResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
##                                        resolutions     = ['et * (sqrt(5.6*5.6/(et*et) + 1.25/et + 0.033))',
##                                                           '0.03  + 1.0/et',
##                                                           '0.015 + 1.5/et'
##                                                           ],
##                                        constraints     =  cms.vdouble(0)
##                                        )

udscResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0334^2 + (1.221/sqrt(et))^2 + (4.7/et)^2))'),
    eta = cms.string('sqrt(0.00809^2 + (0/sqrt(et))^2 + (1.5398/et)^2)'),
    phi = cms.string('sqrt(0.00783^2 + (0/sqrt(et))^2 + (2.581/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0485^2 + (1.146/sqrt(et))^2 + (5.44/et)^2))'),
    eta = cms.string('sqrt(0.00847^2 + (0/sqrt(et))^2 + (1.5396/et)^2)'),
    phi = cms.string('sqrt(0.00691^2 + (0/sqrt(et))^2 + (2.633/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0531^2 + (1.122/sqrt(et))^2 + (5.65/et)^2))'),
    eta = cms.string('sqrt(0.00851^2 + (0/sqrt(et))^2 + (1.5647/et)^2)'),
    phi = cms.string('sqrt(0.00869^2 + (0/sqrt(et))^2 + (2.589/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0436^2 + (1.139/sqrt(et))^2 + (5.6/et)^2))'),
    eta = cms.string('sqrt(0.00809^2 + (0/sqrt(et))^2 + (1.5762/et)^2)'),
    phi = cms.string('sqrt(0.00673^2 + (0/sqrt(et))^2 + (2.627/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0499^2 + (1.111/sqrt(et))^2 + (5.55/et)^2))'),
    eta = cms.string('sqrt(0.00786^2 + (0/sqrt(et))^2 + (1.602/et)^2)'),
    phi = cms.string('sqrt(0.00767^2 + (0/sqrt(et))^2 + (2.62/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0551^2 + (1.081/sqrt(et))^2 + (5.6/et)^2))'),
    eta = cms.string('sqrt(0.00832^2 + (0/sqrt(et))^2 + (1.6007/et)^2)'),
    phi = cms.string('sqrt(0.0057^2 + (0/sqrt(et))^2 + (2.645/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0605^2 + (1.054/sqrt(et))^2 + (5.73/et)^2))'),
    eta = cms.string('sqrt(0.00838^2 + (0/sqrt(et))^2 + (1.5511/et)^2)'),
    phi = cms.string('sqrt(0.00577^2 + (0/sqrt(et))^2 + (2.603/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0563^2 + (1.052/sqrt(et))^2 + (5.83/et)^2))'),
    eta = cms.string('sqrt(0.00882^2 + (0/sqrt(et))^2 + (1.5473/et)^2)'),
    phi = cms.string('sqrt(0.00715^2 + (0/sqrt(et))^2 + (2.569/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0508^2 + (1.119/sqrt(et))^2 + (5.29/et)^2))'),
    eta = cms.string('sqrt(0.00858^2 + (0/sqrt(et))^2 + (1.5522/et)^2)'),
    phi = cms.string('sqrt(0.00657^2 + (0/sqrt(et))^2 + (2.605/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0504^2 + (1.141/sqrt(et))^2 + (5.31/et)^2))'),
    eta = cms.string('sqrt(0.00811^2 + (0/sqrt(et))^2 + (1.618/et)^2)'),
    phi = cms.string('sqrt(0.00631^2 + (0/sqrt(et))^2 + (2.64/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0591^2 + (1.132/sqrt(et))^2 + (5.37/et)^2))'),
    eta = cms.string('sqrt(0.00649^2 + (0.055/sqrt(et))^2 + (1.58/et)^2)'),
    phi = cms.string('sqrt(0.00659^2 + (0/sqrt(et))^2 + (2.612/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0539^2 + (1.136/sqrt(et))^2 + (5.57/et)^2))'),
    eta = cms.string('sqrt(0.00806^2 + (0/sqrt(et))^2 + (1.6097/et)^2)'),
    phi = cms.string('sqrt(0.00654^2 + (0/sqrt(et))^2 + (2.631/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0542^2 + (1.186/sqrt(et))^2 + (5.36/et)^2))'),
    eta = cms.string('sqrt(0.00801^2 + (0/sqrt(et))^2 + (1.6478/et)^2)'),
    phi = cms.string('sqrt(0.00707^2 + (0/sqrt(et))^2 + (2.645/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0583^2 + (1.184/sqrt(et))^2 + (5.41/et)^2))'),
    eta = cms.string('sqrt(0.00834^2 + (0/sqrt(et))^2 + (1.6736/et)^2)'),
    phi = cms.string('sqrt(0.00603^2 + (0/sqrt(et))^2 + (2.681/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0504^2 + (1.235/sqrt(et))^2 + (5.24/et)^2))'),
    eta = cms.string('sqrt(0.00935^2 + (0/sqrt(et))^2 + (1.7398/et)^2)'),
    phi = cms.string('sqrt(0.00738^2 + (0/sqrt(et))^2 + (2.707/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.055^2 + (1.239/sqrt(et))^2 + (5.57/et)^2))'),
    eta = cms.string('sqrt(0.0123^2 + (0/sqrt(et))^2 + (1.773/et)^2)'),
    phi = cms.string('sqrt(0.00726^2 + (0/sqrt(et))^2 + (2.765/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.028^2 + (1.351/sqrt(et))^2 + (5.09/et)^2))'),
    eta = cms.string('sqrt(0.01199^2 + (0/sqrt(et))^2 + (1.784/et)^2)'),
    phi = cms.string('sqrt(0.00808^2 + (0/sqrt(et))^2 + (2.912/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.016^2 + (1.317/sqrt(et))^2 + (5.48/et)^2))'),
    eta = cms.string('sqrt(0.013^2 + (0/sqrt(et))^2 + (1.747/et)^2)'),
    phi = cms.string('sqrt(0.00887^2 + (0/sqrt(et))^2 + (2.924/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.007^2 + (1.228/sqrt(et))^2 + (5.66/et)^2))'),
    eta = cms.string('sqrt(0.00981^2 + (0/sqrt(et))^2 + (1.702/et)^2)'),
    phi = cms.string('sqrt(0.00441^2 + (0/sqrt(et))^2 + (2.78/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0.028^2 + (1.14/sqrt(et))^2 + (5.7/et)^2))'),
    eta = cms.string('sqrt(0.00943^2 + (0/sqrt(et))^2 + (1.777/et)^2)'),
    phi = cms.string('sqrt(0.00804^2 + (0/sqrt(et))^2 + (2.601/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.0274^2 + (1.112/sqrt(et))^2 + (5.48/et)^2))'),
    eta = cms.string('sqrt(0.01049^2 + (0.052/sqrt(et))^2 + (1.723/et)^2)'),
    phi = cms.string('sqrt(0.01001^2 + (0/sqrt(et))^2 + (2.515/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0.0395^2 + (1.007/sqrt(et))^2 + (5.48/et)^2))'),
    eta = cms.string('sqrt(0.01144^2 + (0.075/sqrt(et))^2 + (1.584/et)^2)'),
    phi = cms.string('sqrt(0.01056^2 + (0/sqrt(et))^2 + (2.32/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.0503^2 + (0.82/sqrt(et))^2 + (5.59/et)^2))'),
    eta = cms.string('sqrt(0.01055^2 + (0.041/sqrt(et))^2 + (1.605/et)^2)'),
    phi = cms.string('sqrt(0.0109^2 + (0/sqrt(et))^2 + (2.183/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.0392^2 + (0.828/sqrt(et))^2 + (5.08/et)^2))'),
    eta = cms.string('sqrt(0.01093^2 + (0/sqrt(et))^2 + (1.599/et)^2)'),
    phi = cms.string('sqrt(0.01189^2 + (0/sqrt(et))^2 + (2.056/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.0461^2 + (0.769/sqrt(et))^2 + (4.73/et)^2))'),
    eta = cms.string('sqrt(0.01176^2 + (0/sqrt(et))^2 + (1.49/et)^2)'),
    phi = cms.string('sqrt(0.01082^2 + (0/sqrt(et))^2 + (1.906/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.0436^2 + (0.72/sqrt(et))^2 + (4.2/et)^2))'),
    eta = cms.string('sqrt(0.01413^2 + (0/sqrt(et))^2 + (1.522/et)^2)'),
    phi = cms.string('sqrt(0.01097^2 + (0/sqrt(et))^2 + (1.82/et)^2)'),
    )
    ),
                                        constraints = cms.vdouble(0)
                                        )

bjetResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0897^2 + (1.091/sqrt(et))^2 + (6.01/et)^2))'),
    eta = cms.string('sqrt(0.00475^2 + (0/sqrt(et))^2 + (1.8057/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.255/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0814^2 + (1.218/sqrt(et))^2 + (5.36/et)^2))'),
    eta = cms.string('sqrt(0.00401^2 + (0/sqrt(et))^2 + (1.8376/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.256/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0872^2 + (1.147/sqrt(et))^2 + (5.86/et)^2))'),
    eta = cms.string('sqrt(0.00447^2 + (0/sqrt(et))^2 + (1.8346/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.24/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0834^2 + (1.151/sqrt(et))^2 + (5.75/et)^2))'),
    eta = cms.string('sqrt(0.00434^2 + (0/sqrt(et))^2 + (1.8592/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.269/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0926^2 + (1.085/sqrt(et))^2 + (5.68/et)^2))'),
    eta = cms.string('sqrt(0.00466^2 + (0/sqrt(et))^2 + (1.8458/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.241/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0724^2 + (1.226/sqrt(et))^2 + (5.36/et)^2))'),
    eta = cms.string('sqrt(0.00482^2 + (0/sqrt(et))^2 + (1.8723/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.263/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0697^2 + (1.253/sqrt(et))^2 + (4.94/et)^2))'),
    eta = cms.string('sqrt(0.00566^2 + (0/sqrt(et))^2 + (1.8605/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.201/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0823^2 + (1.095/sqrt(et))^2 + (6.1/et)^2))'),
    eta = cms.string('sqrt(0.00539^2 + (0/sqrt(et))^2 + (1.859/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.26/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.079^2 + (1.171/sqrt(et))^2 + (5.35/et)^2))'),
    eta = cms.string('sqrt(0.00561^2 + (0/sqrt(et))^2 + (1.862/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.225/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0855^2 + (1.141/sqrt(et))^2 + (5.47/et)^2))'),
    eta = cms.string('sqrt(0.00492^2 + (0/sqrt(et))^2 + (1.879/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.226/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0856^2 + (1.173/sqrt(et))^2 + (5.3/et)^2))'),
    eta = cms.string('sqrt(0.00562^2 + (0/sqrt(et))^2 + (1.882/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.219/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.086^2 + (1.199/sqrt(et))^2 + (5.09/et)^2))'),
    eta = cms.string('sqrt(0.00426^2 + (0/sqrt(et))^2 + (1.906/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.248/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0702^2 + (1.326/sqrt(et))^2 + (4.36/et)^2))'),
    eta = cms.string('sqrt(0.00429^2 + (0/sqrt(et))^2 + (1.938/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.256/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0628^2 + (1.406/sqrt(et))^2 + (3.43/et)^2))'),
    eta = cms.string('sqrt(0.00232^2 + (0/sqrt(et))^2 + (2.012/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.263/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0878^2 + (1.235/sqrt(et))^2 + (5.27/et)^2))'),
    eta = cms.string('sqrt(0.00652^2 + (0/sqrt(et))^2 + (2.03/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.335/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0923^2 + (1.255/sqrt(et))^2 + (4.99/et)^2))'),
    eta = cms.string('sqrt(0.00909^2 + (0/sqrt(et))^2 + (2.046/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.462/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.085^2 + (1.327/sqrt(et))^2 + (4.66/et)^2))'),
    eta = cms.string('sqrt(0.00787^2 + (0/sqrt(et))^2 + (2.158/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.632/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.0929^2 + (1.257/sqrt(et))^2 + (4.87/et)^2))'),
    eta = cms.string('sqrt(0.01016^2 + (0/sqrt(et))^2 + (2.111/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.668/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.057^2 + (1.452/sqrt(et))^2 + (3.11/et)^2))'),
    eta = cms.string('sqrt(0.00405^2 + (0/sqrt(et))^2 + (2.118/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.42/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0.0825^2 + (1.222/sqrt(et))^2 + (4.76/et)^2))'),
    eta = cms.string('sqrt(0.0029^2 + (0/sqrt(et))^2 + (2.138/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.223/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.0679^2 + (1.234/sqrt(et))^2 + (4.62/et)^2))'),
    eta = cms.string('sqrt(0.0041^2 + (0/sqrt(et))^2 + (2.158/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (3.018/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0.0651^2 + (1.186/sqrt(et))^2 + (4.32/et)^2))'),
    eta = cms.string('sqrt(0.00454^2 + (0/sqrt(et))^2 + (2.04/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (2.839/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.062^2 + (1.117/sqrt(et))^2 + (4.07/et)^2))'),
    eta = cms.string('sqrt(0.004^2 + (0/sqrt(et))^2 + (1.963/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (2.624/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.1^2 + (1.1/sqrt(et))^2 + (4.1/et)^2))'),
    eta = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (2/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (2.6/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.0707^2 + (0.939/sqrt(et))^2 + (3.91/et)^2))'),
    eta = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (1.7876/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (2.223/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.018^2 + (1.017/sqrt(et))^2 + (3.31/et)^2))'),
    eta = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (1.802/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (2.104/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

metResolution  = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('-3.000<=abs(eta) && abs(eta)<3.000'),
    et  = cms.string('et * (sqrt(0^2 + (1.597/sqrt(et))^2 + (19.37/et)^2))'),
    eta = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0.498/sqrt(et))^2 + (22.468/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )
