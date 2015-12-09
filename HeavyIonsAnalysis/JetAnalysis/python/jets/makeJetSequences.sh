#!/bin/sh

#echo "import FWCore.ParameterSet.Config as cms" > HiGenJetsCleaned_cff.py
#echo "from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *" >> HiGenJetsCleaned_cff.py
echo "from RecoHI.HiJetAlgos.HiGenJets_cff import *" > HiGenJets_cff.py

# ReReco stuff for jec only
echo "import FWCore.ParameterSet.Config as cms" > HiReRecoJets_cff.py
echo "from RecoHI.HiJetAlgos.HiRecoJets_cff import *" >> HiReRecoJets_cff.py
echo "from RecoHI.HiJetAlgos.HiRecoPFJets_cff import *" >> HiReRecoJets_cff.py

for system in PbPb pp
do
    for sample in data mix jec mc
    do
        for algo in ak
        do
            for sub in Vs Pu NONE
            do
                for radius in 1 2 3 4 5 6
                do
                    for object in PF Calo
                    do
                        subt=$sub
                        if [ $sub == "NONE" ]; then
                            subt=""
                        fi
			
                        genjets="HiGenJets"			
                        ismc="False"
                        corrlabel="_offline"
                        domatch="True"
                        genparticles="genParticles"
                        tracks="hiGeneralTracks"
			vertex="offlinePrimaryVertices"
                        pflow="particleFlowTmp"
                        domatch="False"
                        match=""
                        eventinfotag="generator"
			jetcorrectionlevels="\'L2Relative\',\'L3Absolute\'"
                        echo "" > $algo$subt$radius${object}JetSequence_${system}_${sample}_cff.py
			
                        if [ $system == "pp" ]; then
                                #corrlabel="_generalTracks"
                            tracks="generalTracks"
			    vertex="offlinePrimaryVertices"
                            genparticles="genParticles"
                            pflow="particleFlow"
			    if [ $sample == "data" ] && [ $sub == "NONE" ] && [ $radius == 4 ] && [ $object == "PF" ]; then
				jetcorrectionlevels="\'L2Relative\',\'L3Absolute\',\'L2L3Residual\'"
			    fi
                        fi
			
                        if [ $sample == "mc" ] || [ $sample == "jec" ] || [ $sample == "mix" ]; then
                            ismc="True"
                        fi
			
                        if [ $system == "pp" ]; then
                            genjets="HiGenJets"
                        fi
			
                            #if [ $sample == "mix" ]; then
                            #    eventinfotag="hiSignal"
                            #fi
			
                            #if [ $object == "Calo" ]; then
                            #    corrlabel="_HI"
                            #fi
			
                        corrname=`echo ${algo} | sed 's/\(.*\)/\U\1/'`${radius}${object}${corrlabel}
			
                        cat templateSequence_cff.py.txt \
                            | sed "s/ALGO_/$algo/g" \
                            | sed "s/SUB_/$subt/g" \
                            | sed "s/RADIUS_/$radius/g" \
                            | sed "s/OBJECT_/$object/g" \
                            | sed "s/SAMPLE_/$sample/g" \
                            | sed "s/CORRNAME_/$corrname/g" \
                            | sed "s/MATCHED_/$match/g" \
                            | sed "s/ISMC/$ismc/g" \
                            | sed "s/GENJETS/$genjets/g" \
                            | sed "s/GENPARTICLES/$genparticles/g" \
                            | sed "s/TRACKS/$tracks/g" \
                            | sed "s/VERTEX/$vertex/g" \
                            | sed "s/PARTICLEFLOW/$pflow/g" \
                            | sed "s/DOMATCH/$domatch/g" \
                            | sed "s/EVENTINFOTAG/$eventinfotag/g" \
			    | sed "s/JETCORRECTIONLEVELS/$jetcorrectionlevels/g" \
                            >> $algo$subt$radius${object}JetSequence_${system}_${sample}_cff.py
			
# skip no sub
                        if [ $sample == "jec" ] && [ $sub != "NONE" ]; then
                            echo "${algo}${subt}${radius}${object}Jets.jetPtMin = 1" >> HiReRecoJets_cff.py  # NO! this sets ptmin to 1 for all algos
                            if [ $object == "PF" ] && [ $sub != "Pu" ]; then
                                echo "${algo}${subt}${radius}${object}Jets.src = cms.InputTag(\"particleFlowTmp\")" >> HiReRecoJets_cff.py
                            fi
			fi
			if [ $sample == "jec" ]; then
                            echo "${algo}${subt}${radius}${object}JetAnalyzer.genPtMin = cms.untracked.double(1)" >> $algo$subt$radius${object}JetSequence_${system}_${sample}_cff.py
			    echo "${algo}${subt}${radius}${object}JetAnalyzer.jtPtMin = cms.untracked.double(1)" >> $algo$subt$radius${object}JetSequence_${system}_${sample}_cff.py
                        fi
                    done
                done
            done
        done
    done
done

echo "" >> HiGenJets_cff.py
echo "hiGenJets = cms.Sequence(" >> HiGenJets_cff.py

for algo in ak
  do
  for radius in 1 2 3 4 5 6
    do
    echo "$algo${radius}HiGenJets" >> HiGenJets_cff.py
    if [ $radius -ne 6 ]; then
	echo "+" >> HiGenJets_cff.py
    else
	echo ")" >> HiGenJets_cff.py
    fi
    
  done
done

# ReReco stuff, for jec only
echo "" >> HiReRecoJets_cff.py
echo "hiReRecoPFJets = cms.Sequence(" >> HiReRecoJets_cff.py
#echo "PFTowers +" >> HiReRecoJets_cff.py
#echo "voronoiBackgroundPF +" >> HiReRecoJets_cff.py

for sub in NONE Pu Vs
do
    subt=$sub
    if [ $sub == "NONE" ]; then
	subt=""
#   skip no sub
#   fi
    else
#   skip no sub
	for radius in 1 2 3 4 5 6
	do
	    echo "${algo}${subt}${radius}PFJets" >> HiReRecoJets_cff.py
	    if [ $radius -eq 6 ] && [ $sub == "Vs" ]; then
		echo ")" >> HiReRecoJets_cff.py
	    else
		echo "+" >> HiReRecoJets_cff.py
	    fi
	done
#   skip no sub
    fi
#   skip no sub
done

echo "" >> HiReRecoJets_cff.py
echo "hiReRecoCaloJets = cms.Sequence(" >> HiReRecoJets_cff.py
#echo "caloTowersRec*caloTowers*iterativeConePu5CaloJets +" >> HiReRecoJets_cff.py
#echo "voronoiBackgroundCalo +" >> HiReRecoJets_cff.py

for sub in NONE Pu Vs
do
    subt=$sub
    if [ $sub == "NONE" ]; then
	subt=""
#   skip no sub
#    fi
    else
#   skip no sub
	for radius in 1 2 3 4 5 6
	do
	    echo "${algo}${subt}${radius}CaloJets" >> HiReRecoJets_cff.py
	    if [ $radius -eq 6 ] && [ $sub == "Vs" ]; then
		echo ")" >> HiReRecoJets_cff.py
	    else
		echo "+" >> HiReRecoJets_cff.py
	    fi
	done
#   skip no sub
    fi
#   skip no sub
done

cat HiReRecoJets_cff.py | sed "s/particleFlowTmp/particleFlow/g" > HiReRecoJets_pp_cff.py
mv HiReRecoJets_cff.py HiReRecoJets_HI_cff.py
