#!/bin/sh

echo "import FWCore.ParameterSet.Config as cms" > HiGenJetsCleaned_cff.py
echo "from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *" >> HiGenJetsCleaned_cff.py

# ReReco stuff for jec only
echo "import FWCore.ParameterSet.Config as cms" > HiReRecoJets_cff.py
echo "from RecoHI.HiJetAlgos.HiRecoJets_cff import *" >> HiReRecoJets_cff.py
echo "from RecoHI.HiJetAlgos.HiRecoPFJets_cff import *" >> HiReRecoJets_cff.py

for system in PbPb pp pPb
do
    for sample in data mc mix jec
    do
        for algo in ak
        do
            for sub in Vs Pu NONE
            do
                for radius in 1 2 3 4 5 6 7
                do
                    for object in PF Calo
                    do
                        for btaggers in bTag_ NONE
                        do

                            subt=$sub
                            if [ $sub == "NONE" ]; then
                                subt=""
                            fi

                            if [ $btaggers == "NONE" ]; then
                                btag=""
                                genjets="HiGenJetsCleaned"
                            elif [ $system == "pPb" ]; then
                                btag=$btaggers
                                genjets="HiGenJets"
                            else
                                btag=$btaggers
                                genjets="HiGenJetsCleaned"
                            fi

                            ismc="False"
                            corrlabel="_hiIterativeTracks"
                            domatch="True"
                            genparticles="hiGenParticles"
                            tracks="hiGeneralTracks"
                            pflow="particleFlowTmp"
                            domatch="False"
                            match=""
                            eventinfotag="generator"
                            echo "" > $algo$subt$radius${object}JetSequence_${system}_${sample}_${btag}cff.py

                            if [ $system == "pPb" ]; then
                                corrlabel="_generalTracks"
                                tracks="generalTracks"
                                genparticles="hiGenParticles"
                                pflow="particleFlow"
                            fi

                            if [ $system == "pp" ]; then
                                corrlabel="_generalTracks"
                                tracks="generalTracks"
                                genparticles="genParticles"
                                pflow="particleFlow"
                            fi

                            if [ $sample == "mc" ] || [ $sample == "jec" ] || [ $sample == "mix" ]; then
                                ismc="True"
                            fi

                            if [ $system == "pp" ]; then
                                genjets="HiGenJets"
                            fi

                            if [ $sample == "mix" ]; then
                                eventinfotag="hiSignal"
                            fi

                            if [ $object == "Calo" ]; then
                                corrlabel="_HI"
                            fi

                            corrname=`echo ${algo} | sed 's/\(.*\)/\U\1/'`${subt}${radius}${object}${corrlabel}

                            if [ $system == "PbPb" ] && [ $sample == "mc" ] && [ $object == "PF" ] && [ $sub == "Vs" ] && [ $btaggers == "NONE" ] ; then

                                cat templateClean_cff.py.txt \
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
                                | sed "s/PARTICLEFLOW/$pflow/g" \
                                | sed "s/DOMATCH/$domatch/g" \
                                >> HiGenJetsCleaned_cff.py
                            fi

                            if [ $btaggers == "bTag_" ]; then
                                cat templateSequence_bTag_cff.py.txt \
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
                                | sed "s/PARTICLEFLOW/$pflow/g" \
                                | sed "s/DOMATCH/$domatch/g" \
                                | sed "s/EVENTINFOTAG/$eventinfotag/g" \
                                >> $algo$subt$radius${object}JetSequence_${system}_${sample}_${btag}cff.py
                            fi
                            if [ $btaggers == "NONE" ]; then
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
                                | sed "s/PARTICLEFLOW/$pflow/g" \
                                | sed "s/DOMATCH/$domatch/g" \
                                | sed "s/EVENTINFOTAG/$eventinfotag/g" \
                                >> $algo$subt$radius${object}JetSequence_${system}_${sample}_cff.py
                            fi

                            if [ $sample == "jec" ] && [ $btaggers == "NONE" ]; then
                                echo "${algo}${subt}${radius}${object}Jets.jetPtMin = 1" >> HiReRecoJets_cff.py
                                if [ $object == "PF" ] && [ $sub != "Pu" ]; then
                                    echo "${algo}${subt}${radius}${object}Jets.src = cms.InputTag(\"particleFlowTmp\")" >> HiReRecoJets_cff.py
                                fi
			    fi
			    if [ $sample == "jec" ]; then
                                echo "${algo}${subt}${radius}${object}JetAnalyzer.genPtMin = cms.untracked.double(1)" >> $algo$subt$radius${object}JetSequence_${system}_${sample}_${btag}cff.py
                            fi
                        done
                    done
                done
            done
        done
    done
done

echo "" >> HiGenJetsCleaned_cff.py
echo "hiGenJetsCleaned = cms.Sequence(" >> HiGenJetsCleaned_cff.py

for algo in ak
  do
  for radius in 1 2 3 4 5 6 7
    do
    echo "$algo${radius}HiGenJetsCleaned" >> HiGenJetsCleaned_cff.py
    if [ $radius -ne 7 ]; then
	echo "+" >> HiGenJetsCleaned_cff.py
    else
	echo ")" >> HiGenJetsCleaned_cff.py
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
    fi
    for radius in 1 2 3 4 5 6 7
    do
	echo "${algo}${subt}${radius}PFJets" >> HiReRecoJets_cff.py
	if [ $radius -eq 7 ] && [ $sub == "Vs" ]; then
	    echo ")" >> HiReRecoJets_cff.py
	else
	    echo "+" >> HiReRecoJets_cff.py
	fi
    done
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
    fi
    for radius in 1 2 3 4 5 6 7
    do
	echo "${algo}${subt}${radius}CaloJets" >> HiReRecoJets_cff.py
	if [ $radius -eq 7 ] && [ $sub == "Vs" ]; then
	    echo ")" >> HiReRecoJets_cff.py
	else
	    echo "+" >> HiReRecoJets_cff.py
	fi
    done
done

cat HiReRecoJets_cff.py | sed "s/particleFlowTmp/particleFlow/g" > HiReRecoJets_pp_cff.py
mv HiReRecoJets_cff.py HiReRecoJets_HI_cff.py
