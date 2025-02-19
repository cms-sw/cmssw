#!/bin/sh

castorhome="/castor/cern.ch/user/p/pjanot/CMSSW160pre5/"

# for energy in 1 2 3 4 5 7 9 12 15 20 30 50 100 200 300 500 700 1000
for energy in 30 50 100 200
  do
  echo "Energy "$energy
  
  for pid in 211 -211 130 321 -321 2212 -2212 2112 -2112
    do
#    echo "PID "$pid
    
    case $energy in
	1)
	    maxnu=300000
	    totev=300000
	    ;;
	2)
	    maxnu=300000
	    totev=300000
	    ;;
	3)
	    maxnu=300000
	    totev=300000
	    ;;
	4)
	    maxnu=300000
	    totev=300000
	    ;;
	5)
	    maxnu=300000
	    totev=300000
	    ;;
	7)
	    maxnu=300000
	    totev=300000
	    ;;
	9)
	    maxnu=300000
	    totev=300000
	    ;;
	12)
	    maxnu=200000
	    totev=300000
	    ;;
	15)
	    maxnu=150000
	    totev=300000
	    ;;
	20)
	    maxnu=150000
	    totev=300000
	    ;;
	30)
	    maxnu=150000
	    totev=150000
	    ;;
	50)
	    maxnu=100000
	    totev=150000
	    ;;
	100)
	    maxnu=50000
	    totev=150000
	    ;;
	200)
	    maxnu=25000
	    totev=150000
	    ;;
	300)
	    maxnu=10000
	    totev=150000
	    ;;
	500)
	    maxnu=10000
	    totev=150000
	    ;;
	700)
	    maxnu=10000
	    totev=150000
	    ;;
	1000)
	    maxnu=10000
	    totev=150000
	    ;;
	*)
	    maxnu=0
	    totev=0
	    ;;
    esac

    case $pid in
	211)
	    maxnu=$(( maxnu*1 ))
	    part="piplus"
	    ;;
	-211)
	    maxnu=$(( maxnu*1 ))
	    part="piminus"
	    ;;
	130)
	    maxnu=$(( maxnu/4 ))
	    part="K0L"
	    ;;
	321)
	    maxnu=$(( maxnu/5 ))
	    part="Kplus"
	    ;;
	-321) 
	    maxnu=$(( maxnu/5 ))
	    part="Kminus"
	    ;;
	2212) 
	    maxnu=$(( maxnu*15 /100 ))
	    part="p"
	    ;;
	-2212)
	    maxnu=$(( maxnu*15 /100 ))
	    part="pbar"
	    ;;
	2112)
	    maxnu=$(( maxnu*15 /100 ))
	    part="n"
	    ;;
	-2112)
	    maxnu=$(( maxnu*15 /100 ))
	    part="nbar"
	    ;;
	*)
	    maxnu=$(( maxnu*0 ))
	    ;;
    esac

    if (( maxnu < 10000 )); then
	maxnu=10000
    fi

    if (( energy == 1 )); then
	if (( pid == 2212 || pid == -2212 || pid == 2112 || pid == -2112 )); then
	    ene="1.4"
	else
	    ene=$energy
	fi
    else
	ene=$energy
    fi

    DIRNAME=$castorhome"SingleParticlePID"$pid"-E"$energy
    NAME="SingleParticlePID"$pid"-E"$ene
    
#    echo "Name "$NAME
    inputfilename=$NAME
#    echo $inputfilename	

    File1=$DIRNAME"/"$NAME"_0.root"
    File2=$DIRNAME"/"$NAME"_1.root"
    File3=$DIRNAME"/"$NAME"_2.root"
    File4=$DIRNAME"/"$NAME"_3.root"
    File5=$DIRNAME"/"$NAME"_4.root"

#    echo "File1 "$File1
#    echo "File2 "$File2
#    echo "File3 "$File3
#    echo "File4 "$File4
#    echo "File5 "$File5
    
    nufile="NuclearInteractions_"$part"_E"$energy".root"
    rootfile=$part"_E"$energy".root"
    txtfile=$part"_E"$energy".txt"

#    echo "MAXNU" $maxnu
    
 
    sed -e "s/==INPUTFILE==/$inputfilename/" -e "s/==TOTEV==/$totev/" -e "s/==MAXNU==/$maxnu/" -e "s/==NUFILE==/$nufile/" -e "s/==ROOTFILE==/$rootfile/" NITemplate_cfg.py > tmp_pyfile
#    
#Start to write the script
    cat > job_${NAME}.sh << EOF
#!/bin/sh
scramv1 project CMSSW CMSSW_1_6_0_pre6
cd CMSSW_1_6_0_pre6
cd src/
mkdir FastSimulation
cd FastSimulation
mkdir MaterialEffects
cd MaterialEffects
mkdir test
cd test
cp /afs/cern.ch/user/p/pjanot/scratch0/CMSSW_1_6_0_pre6/src/FastSimulation/MaterialEffects/test/*.cc . 
cp /afs/cern.ch/user/p/pjanot/scratch0/CMSSW_1_6_0_pre6/src/FastSimulation/MaterialEffects/test/BuildFile . 
cd ../../
cp -r /afs/cern.ch/user/p/pjanot/scratch0/CMSSW_1_6_0_pre6/src/FastSimulation/EventProducer .
cd ..
scramv1 b -j 4
eval \`scramv1 runtime -sh\`
rfcp $File1 .
rfcp $File2 .
rfcp $File3 .
rfcp $File4 .
rfcp $File5 .
#
#commande pour decoder le .py
cat > Rereco_cfg.py << "EOF"
EOF
    
#Ajoute le .py au script
cat tmp_pyfile >> job_${NAME}.sh
#
# On poursuit le script
echo "EOF" >> job_${NAME}.sh
cat >> job_${NAME}.sh << EOF
cmsRun Rereco_cfg.py > $txtfile

rfcp $nufile $DIRNAME/$nufile
rfcp $rootfile $DIRNAME/$rootfile
cp $nufile /afs/cern.ch/cms/data/CMSSW/FastSimulation/MaterialEffects/data/$nufile
cp $rootfile /afs/cern.ch/cms/data/CMSSW/FastSimulation/MaterialEffects/data/$rootfile
cp $txtfile /afs/cern.ch/cms/data/CMSSW/FastSimulation/MaterialEffects/data/$txtfile
rm *.root
rm *.txt

EOF
chmod 755 job_${NAME}.sh
bsub -q 1nh -J $NAME $PWD/job_${NAME}.sh
#
    
  done
done
