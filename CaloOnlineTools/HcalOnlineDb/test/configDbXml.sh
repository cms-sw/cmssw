#!/bin/bash
echo ''
echo ''
echo '========================================================='
echo '='
echo '= This script prepares HCAL configuration data'
echo '= for uploading to the Oracle configuration database'
echo '='
echo '=    written by: Gena Kukartsev'
echo '=    email:      kukarzev@fnal.gov'
echo '='
echo '=    March 5, 2008'
echo '='
echo '========================================================='

lutMenu()
{
  echo ''
  echo '  -- Processing trigger lookup tables'
  echo ''
  echo -n 'Please enter path to the directory with the LUT'
  echo 'and checksums XML files.'
  echo -n 'LUT file names are expected to end with _X.xml '
  echo 'or _XX.xml where XX is the crate number.'
  echo -n 'LUT checksums file name is expected to end '
  echo 'with _checksums.xml, and should be just one file.'
  echo ''
  echo -n 'Path: '
  read lut_path_temp
  lut_path=`echo ${lut_path_temp%/}/`
  echo ''
  echo 'processing LUTs from' $lut_path '...'
  lut_files_num=`find $lut_path -iname "*_[0-9]*.xml" | wc -l`
  lutchecksums_files_num=`find $lut_path -iname "*_checksums*.xml" | wc -l`
  echo $lut_files_num 'LUT files found...'
  echo $lutchecksums_files_num 'LUT checksums files found...'
  if [ $lut_files_num -gt 0 ]
  then
#      echo 'found LUTs XML... '
      echo -n 'creating temp directory... '
      lut_temp_dir=`mktemp -d`
      echo $lut_temp_dir
      cp `find $lut_path -iname "*_[0-9]*.xml"` $lut_temp_dir/
      cp `find $lut_path -iname "*_checksums*.xml"` $lut_temp_dir/
      a_lut_file=`find $lut_temp_dir/ -iname "*_[0-9]*.xml" | awk 'NR==1{print $1}'`
      source luts.sh $a_lut_file
      echo -n 'Cleaning temporary files... '
      rm -rf $lut_temp_dir
      echo 'done'
  else
      echo 'LUT XML not found...check path, leading-trailing / etc...'
  fi
}

credits()
{
    echo ''
    echo 'Questions regarding this script can be forwarded to'
    echo '  Gena Kukartsev, email: kukarzev@fnal.gov'
    echo ''
    echo -n 'Questions regarding the database operations should '
    echo 'be forwarded to'
    echo '  Gennadiy Lukhanin, email: lukhanin@fnal.gov'
    echo ''
    echo -n 'However, CMS TWiki is your friend. Many answers to '
    echo 'the database questions you will find at'
    echo 'https://twiki.cern.ch/twiki/bin/view/CMS/CMSHCALConfigDBDesign'
}

mainMenu()
{
  echo ''
  echo '  -- Main menu'
  echo ' 1. Trigger lookup tables'
  echo ' 2. Contact info'
  
  echo ''
  echo -n 'Please choose the configuration type: '
  read line
#  echo $line
  echo ''

  case $line in
      1)
        lutMenu
	;;
      2)
      credits
      ;;
      *)
        echo 'Invalid choice - nothing to do...'
        credits
      ;;
  esac

}

mainMenu
