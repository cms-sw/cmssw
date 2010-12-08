INPUT=benchmark-simple.txt
#(combine $INPUT -M ProfileLikelihood > $INPUT.log.ProfileLikelihood 2>&1 &)
#(combine $INPUT -M BayesianFlatPrior > $INPUT.log.BayesianFlatPrior 2>&1 &)
#(combine $INPUT -M BayesianFlatPrior --prior '1/sqrt(r)' > $INPUT.log.BayesianFlatPrior.SqrtR 2>&1 &)
(combine $INPUT -M MarkovChainMC --proposal uniform -i 200000 -H ProfileLikelihood > $INPUT.log.MCMC.uniform 2>&1 &)
(combine $INPUT -M MarkovChainMC --proposal gaus    -i 200000 -H ProfileLikelihood > $INPUT.log.MCMC.gaus    2>&1 &)
#(combine $INPUT -M Hybrid --rule CLs       -H ProfileLikelihood > $INPUT.log.CLs.LEP  2>&1 &)
#(combine $INPUT -M Hybrid --rule CLsplusb  -H ProfileLikelihood > $INPUT.log.CLsb.LEP 2>&1 &)

