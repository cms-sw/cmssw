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

foreach gtag ( STARTUP MC )

  foreach table ( GRun HIon )

    foreach task ( RelVal_DigiL1Raw )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}

      if ( $table == HIon ) then
#        cat >> $name.py <<EOF
## override the L1 menu
#if 'GlobalTag' in process.__dict__:
#    if not 'toGet' in process.GlobalTag.__dict__:
#        process.GlobalTag.toGet = cms.VPSet( )
#    process.GlobalTag.toGet.append(
#        cms.PSet(  
#            record  = cms.string( "L1GtTriggerMenuRcd" ),
#            tag     = cms.string( "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v0_mc" ),
#            connect = cms.untracked.string( "sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_CollisionsHeavyIons2010_v0_mc.db" )
#        )
#    )
#EOF
      endif

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

    foreach task ( RelVal_HLT RelVal_HLT2 )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}

      if ( $table == HIon ) then
#        cat >> $name.py <<EOF
## override the L1 menu
#if 'GlobalTag' in process.__dict__:
#    if not 'toGet' in process.GlobalTag.__dict__:
#        process.GlobalTag.toGet = cms.VPSet( )
#    process.GlobalTag.toGet.append(
#        cms.PSet(  
#            record  = cms.string( "L1GtTriggerMenuRcd" ),
#            tag     = cms.string( "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v0_mc" ),
#            connect = cms.untracked.string( "sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_CollisionsHeavyIons2010_v0_mc.db" )
#        )
#    )
#EOF
      endif

      echo "cmsRun $name.py >& $name.log"
#     ls -l        $name.py
      time  cmsRun $name.py >& $name.log
      echo "exit status: $?"

    end

  end

end

# separate reco task to run last

foreach gtag ( STARTUP )

  foreach table ( GRun HIon )

    foreach task ( RelVal_HLT_RECO )

      echo
      set name = ${task}_${table}_${gtag}
      rm -f $name.{log,root}

      if ( $table == HIon ) then
#        cat >> $name.py <<EOF
## override the L1 menu
#if 'GlobalTag' in process.__dict__:
#    if not 'toGet' in process.GlobalTag.__dict__:
#        process.GlobalTag.toGet = cms.VPSet( )
#    process.GlobalTag.toGet.append(
#        cms.PSet(  
#            record  = cms.string( "L1GtTriggerMenuRcd" ),
#            tag     = cms.string( "L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v0_mc" ),
#            connect = cms.untracked.string( "sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_CollisionsHeavyIons2010_v0_mc.db" )
#        )
#    )
#EOF
      endif

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
