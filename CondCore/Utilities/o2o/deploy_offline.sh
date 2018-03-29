#!/bin/sh                                                                                                                                                                 

home=~
localhome=/data/condbpro
root=/data/O2O
cmsswroot=/cvmfs/cms.cern.ch/
extroot=/data/ext
release=CMSSW_10_0_5
arch=slc6_amd64_gcc630
sourceroot=CondCore/Utilities/o2o/templates
source=$cmsswroot/$arch/cms/cmssw/$release/src/$sourceroot

files=( setup.sh
        setStrip.sh
        SiStripDCS.sh
	ecal_laser.sh
	EcalLaser.sh
	EcalLaserTest.sh
	EcalLaser_express.sh
	EcalLaser_expressTest.sh )

folders=( EcalLaser
          EcalLaserTest
          EcalLaser_express
          EcalLaser_expressTest
	  SiStrip )

cd $root
if [ ! -d scripts ]; then
    mkdir scripts
fi
if [ ! -d logs ]; then
    mkdir logs
fi

sed_fmt () {
  var=$(echo $1 | sed -e "s#/#\\\/#g")
}

replace_params () {
  params=( @root
           @home
	   @cmsswroot
           @extroot )
  tgt_file=$1
  var=''
  # replace path params
  sed_fmt $root
  tgt_root=$var
  sed -i -e s/@root/$tgt_root/g $tgt_file
  sed_fmt $cmsswroot
  tgt_cmsswroot=$var
  sed -i -e s/@cmsswroot/$tgt_cmsswroot/g $tgt_file
  sed_fmt $extroot
  tgt_extroot=$var
  sed -i -e s/@extroot/$tgt_extroot/g $tgt_file
  sed_fmt $home
  tgt_home=$var
  # relace non-path params
  sed -i -e s/@home/$tgt_home/g $tgt_file
  sed -i -e s/@release/$release/g $tgt_file
  sed -i -e s/@arch/$arch/g $tgt_file
}

for file in "${files[@]}"
do
    cp $source/$file scripts/
    tgt_file=scripts/$file
    replace_params $tgt_file
done
for f in "${folders[@]}"
do  
    if [ ! -d $f ]; then
	mkdir -p $f
    fi
    if [ ! -d logs/$f ]; then
	mkdir logs/$f
    fi
done
