#!/bin/tcsh

cmsenv

rehash

echo
echo "Existing cfg files:"
ls -l OnData*.py
ls -l OnLine*.py

echo
echo "Creating offline configs with cmsDriver"
echo "./cmsDriver.sh"
time  ./cmsDriver.sh

echo
echo "Running selected configs from:"
pwd

foreach gtag ( STARTUP )
#foreach gtag ( STARTUP MC )

  foreach table ( GRun HIon )

    foreach task ( RelVal_DigiL1Raw )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}

#      cat >> $name.py <<EOF
## load 4.2.x JECs
#if 'GlobalTag' in process.__dict__:
#    if not 'toGet' in process.GlobalTag.__dict__:
#        process.GlobalTag.toGet = cms.VPSet( )
#    process.GlobalTag.toGet.append(
#        cms.PSet(
#            record  = cms.string( 'JetCorrectionsRecord' ),
#            tag     = cms.string( 'JetCorrectorParametersCollection_Jec11_V1_AK5Calo' ),
#            label   = cms.untracked.string( 'AK5Calo' ),
#            connect = cms.untracked.string( 'frontier://PromptProd/CMS_COND_31X_PHYSICSTOOLS' )
#        )
#    )
#EOF

      echo "cmsRun $name.py >& $name.log"
#     ls -l        $name.py
      time  cmsRun $name.py >& $name.log
      echo "exit status: $?"

    end

    if ( $gtag == STARTUP ) then

#     link to input file for subsequent OnLine* step
      rm -f RelVal_DigiL1Raw_${table}.root
      ln -s RelVal_DigiL1Raw_${table}_${gtag}.root RelVal_DigiL1Raw_${table}.root
      foreach task ( OnData_HLT OnLine_HLT )

        echo
        set name = ${task}_${table}
        rm -f $name.{log,root}

        echo "cmsRun $name.py >& $name.log"
#       ls -l        $name.py
        time  cmsRun $name.py >& $name.log
        echo "exit status: $?"

      end

    endif

    foreach task ( RelVal_HLT RelVal_HLT2 FastSim_GenToHLT )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}

#      cat >> $name.py <<EOF
## load 4.2.x JECs
#if 'GlobalTag' in process.__dict__:
#    if not 'toGet' in process.GlobalTag.__dict__:
#        process.GlobalTag.toGet = cms.VPSet( )
#    process.GlobalTag.toGet.append(
#        cms.PSet(
#            record  = cms.string( 'JetCorrectionsRecord' ),
#            tag     = cms.string( 'JetCorrectorParametersCollection_Jec11_V1_AK5Calo' ),
#            label   = cms.untracked.string( 'AK5Calo' ),
#            connect = cms.untracked.string( 'frontier://PromptProd/CMS_COND_31X_PHYSICSTOOLS' )
#        )
#    )
#EOF

      if ($table == HIon && $task == FastSim_GenToHLT ) then
      echo "$name does not exist!"
      else
      echo "cmsRun $name.py >& $name.log"
#     ls -l        $name.py
      time  cmsRun $name.py >& $name.log
      echo "exit status: $?"
      endif

    end

  end

end

# special fastsim tests

# foreach task ( IntegrationTestWithHLT_cfg ExampleWithHLT_GRun_cfg )
foreach task ( IntegrationTestWithHLT_cfg )

  echo
  set name = ${task}
  rm -f $name.{log,root}

  echo "cmsRun $CMSSW_BASE/src/FastSimulation/Configuration/test/$name.py >& $name.log"
# ls -l        $CMSSW_BASE/src/FastSimulation/Configuration/test/$name.py
  time  cmsRun $CMSSW_BASE/src/FastSimulation/Configuration/test/$name.py >& $name.log
  echo "exit status: $?"

end

# separate hlt+reco tasks to run last

foreach gtag ( STARTUP )

  foreach table ( GRun HIon )

    foreach task ( RelVal_HLT_RECO )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}

#      cat >> $name.py <<EOF
## load 4.2.x JECs
#if 'GlobalTag' in process.__dict__:
#    if not 'toGet' in process.GlobalTag.__dict__:
#        process.GlobalTag.toGet = cms.VPSet( )
#    process.GlobalTag.toGet.append(
#        cms.PSet(
#            record  = cms.string( 'JetCorrectionsRecord' ),
#            tag     = cms.string( 'JetCorrectorParametersCollection_Jec11_V1_AK5Calo' ),
#            label   = cms.untracked.string( 'AK5Calo' ),
#            connect = cms.untracked.string( 'frontier://PromptProd/CMS_COND_31X_PHYSICSTOOLS' )
#        )
#    )
#EOF

      echo "cmsRun $name.py >& $name.log"
#     ls -l        $name.py
      time  cmsRun $name.py >& $name.log
      echo "exit status: $?"

    end

  end

end

echo
echo "Resulting log files:"
ls -lt *.log
