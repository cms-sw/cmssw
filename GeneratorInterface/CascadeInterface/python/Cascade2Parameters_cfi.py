import FWCore.ParameterSet.Config as cms

                         
Cascade2Parameters = cms.PSet(
    parameterSets = cms.vstring('CascadeSettings'),
    CascadeSettings = cms.vstring('NCB = 50000   ! number of calls per iteration for bases',
                                  'ACC1 = 1.0    ! relative precision for grid optimisation',
                                  'ACC2 = 0.5    ! relative precision for integration',
                                  'KE = 2212     ! flavour code of beam1',
                                  'IRES(1) = 1   ! direct or resolved particle 1',
                                  'KP = 2212     ! flavour code of beam2',
                                  'IRES(2) = 1   ! direct or resolved particle 2',
                                  'NFLAV = 5     ! number of active flavors',
                                  'IPRO = 10     ! hard subprocess number',
                                  'IRPA = 1      ! IPRO = 10, switch to select QCD process g* g* -> q qbar',
                                  'IRPB = 1      ! IPRO = 10, switch to select QCD process g* g -> g g',
                                  'IRPC = 1      ! IPRO = 10, switch to select QCD process g* q -> g q',
                                  'IHFLA = 4     ! flavor of heavy quark produced (IPRO = 11, 504, 514)',
                                # 'I23S = 0      ! select vector meson state (0 = 1S, 2 = 2S, 3 = 3S from version 2.2.03 on)',
                                  'IPSIPOL = 0   ! use matrix element including J/psi (Upsilon) polarisation (1 = on, 0 = off)',
                                  'PT2CUT = 0.0  ! pt2 cut in ME for massless partons'
                                  'NFRAG = 1     ! switch for fragmentation (1 = on, 0 = off)',
                                  'IFPS = 3      ! switch for parton shower: (0 = off - 1 = initial - 2 = final - 3 = initial & final)',
                                  'ITIMSHR = 1   ! switch for time like parton shower in intial state (0 = off,1 = on)',
                                  'ICCFM = 1     ! select CCFM or DGLAP evolution (CCFM = 1, DGLAP = 0)'
                                  'IFINAL = 1    ! scale switch for final state parton shower',
                                  'SCALFAF = 1.0 ! scale factor for final state parton shower',
                                  'IRspl = 4     ! switch for proton remnant treatment',
                                  'IPST = 0      ! keep track of intermediate state in parton shower ',
                                  'INTER = 0     ! mode of interaction for ep (photon exchange, Z-echange (not implemented))',
                                  'IRUNAEM = 1   ! switch for running alphaem (0 = off,1 = on)',
                                  'IRUNA = 1     ! switch for running alphas (0 = off,1 = on)', 
                                  'IQ2 = 3       ! scale in alphas',
                                  'SCALFA = 1.0  ! scale factor for scale in alphas',
                                  'IGLU = 1010   ! select uPDF'))
