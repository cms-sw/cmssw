#!/bin/sh

castorDir=/castor/cern.ch/user/a/azzi/CMSSW360pre2/
cmsswDir=/afs/cern.ch/user/a/azzi/scratch0/FastSimTrackTuning/CMSSW_3_6_0_pre2/src/

njobs=0
nevt=0
#for ((energy=0; energy<=27; energy++));
#1 -> 23 faits 
#24->27 fait

 for ((energy=0; energy<=7; energy++));
  do
  case $energy in
      0)
	  ptmin=0
	  ptmax=1
	  jobs=1
	  ;;
      1)
	  ptmin=1
	  ptmax=2
	  jobs=1
	  ;;
      2)
	  ptmin=2
	  ptmax=3
	  jobs=1
	  ;;
      3)
	  ptmin=3
	  ptmax=4
	  jobs=2
	  ;;
      4)
	  ptmin=4
	  ptmax=5
	  jobs=2
	  ;;
      5)
	  ptmin=5
	  ptmax=6
	  jobs=4
	  ;;
      6)
	  ptmin=6
	  ptmax=8
	  jobs=4
	  ;;
      7)
	  ptmin=8
	  ptmax=10
	  jobs=4
	  ;;
      8)
	  ptmin=10
	  ptmax=12
	  jobs=2
	  ;;
      9)
	  ptmin=12
	  ptmax=15
	  jobs=2
	  ;;
      10)
	  ptmin=15
	  ptmax=20
	  jobs=2
	  ;;
      11)
	  ptmin=20
	  ptmax=25
	  jobs=2
	  ;;
      12)
	  ptmin=25
	  ptmax=30
	  jobs=2
	  ;;
      13)
	  ptmin=30
	  ptmax=40
	  jobs=2
	  ;;
      14)
	  ptmin=40
	  ptmax=50
	  jobs=2
	  ;;
      15)
	  ptmin=50
	  ptmax=60
	  jobs=2
	  ;;
      16)
	  ptmin=60
	  ptmax=80
	  jobs=2
	  ;;
      17)
	  ptmin=80
	  ptmax=100
	  jobs=2
	  ;;
      18)
	  ptmin=100
	  ptmax=120
	  jobs=4
	  ;;
      19)
	  ptmin=120
	  ptmax=150
	  jobs=4
	  ;;
      20)
	  ptmin=150
	  ptmax=200
	  jobs=4
	  ;;
      21)
	  ptmin=200
	  ptmax=250
	  jobs=5
	  ;;
      22)
	  ptmin=250
	  ptmax=300
	  jobs=5
	  ;;
      23)
	  ptmin=300
	  ptmax=400
	  jobs=8
	  ;;
      24)
	  ptmin=400
	  ptmax=500
	  jobs=8
	  ;;
      25)
	  ptmin=500
	  ptmax=600
	  jobs=8
	  ;;
      26)
	  ptmin=600
	  ptmax=800
	  jobs=8
	  ;;
      27)
	  ptmin=800
	  ptmax=1000
	  jobs=8
	  ;;
    esac
	
    numevt=$(( 10000 /($jobs) ))
    
  for ((job=1; job<=jobs; job++));
  do

    njobs=$(( njobs+1 ))
    nevt=$(( nevt+numevt ))
    echo "JOB "$njobs" : "$job", Energy = "$energy", "$nevt" events"
      name="SinglePion_E"$energy"_"${job}
      filename="fevt_"${name}".root"
      logname="log_"${name}".txt"
      echo $name
      
      seed1=$(( ($job+1) + 10985*($energy+1) ))
      sed -e "s/==seed1==/${seed1}/" -e "s/==MINPT==/${ptmin}/" -e "s/==MAXPT==/${ptmax}/" -e "s/==NUMEVT==/${numevt}/" SinglePionFullSim_cfg.py > tmp_cfg
      
#Start to write the script
      cat > job_${name}.sh << EOF

#!/bin/sh
cd $cmsswDir
eval \`scramv1 runtime -sh\`
cd -
#commande pour decoder le .cfg
cat > TEST_cfg.py << "EOF"
EOF
      
#Ajoute le .cfg au script
cat tmp_cfg >> job_${name}.sh

# On poursuit le script
echo "EOF" >> job_${name}.sh
cat >> job_${name}.sh << EOF


cmsRun TEST_cfg.py >& log

rfcp fevt.root $castorDir$filename

EOF
chmod 755 job_${name}.sh
bsub -q cmst3 -R "mem>2000" -J $name $PWD/job_${name}.sh


  done
done
