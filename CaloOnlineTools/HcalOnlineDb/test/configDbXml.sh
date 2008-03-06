#!/bin/bash
echo ''
echo ''
echo '========================================================='
echo '='
echo '= This script prepares HCAL configuration data'
echo '= for uploading to the Oracle configuration database'
echo '='
echo '=    written by: Gena Kukartsev,'
echo '=                Brown University'
echo '='
echo '=    email:      kukarzev@fnal.gov'
echo '='
echo '=    March 5, 2008'
echo '='
echo '=        Fear is the path to the Dark Side....'
echo '='
echo '========================================================='

compileMessage()
{
    echo ''
    echo 'ERROR!'
    echo ''
    echo -n 'the xmlToolsRun executable is missing. '
    echo 'Please compile the package. Issue gmake in'
    echo 'the current directory.'
    echo ''
}

uploadInstructions()
{
echo ''
echo -n $1
echo -n ' are prepared for uploading to OMDS and saved in '
echo $2
echo ''
echo 'REMEMBER!'
echo -n 'It is always a good idea to upload to the validation '
echo 'database first before uploading to OMDS'
echo ''
echo -n 'In order to upload to a database, copy '
echo -n $2
echo ' to'
echo 'dbvalhcal@pcuscms34.cern.ch:conditions/ (validation - first!)'
echo 'dbpp5hcal@pcuscms34.cern.ch:conditions/ (OMDS)'
echo ''
echo -n 'or, even better, follow the most recent instructions at '
echo 'https://twiki.cern.ch/twiki/bin/view/CMS/OnlineHCALDataSubmissionProceduresTOProdOMDSP5Server'
echo ''
}

zsMenu()
{
    echo ''
    echo -n 'Enter desired tag name: '
    read tag_name
    echo -n 'Enter comment: '
    read comment
    echo -n 'Enter zero suppression for HB: '
    read hb_value
    echo -n 'Enter zero suppression for HE: '
    read he_value
    echo -n 'Enter zero suppression for HO: '
    read ho_value
    echo -n 'Enter zero suppression for HF: '
    read hf_value
    echo 'Creating HTR Zero Suppression values...'
    ./xmlToolsRun --zs2 --tag=$tag_name --comment="$comment" --zs2HB="$hb_value" --zs2HE="$he_value" --zs2HO="$ho_value" --zs2HF="$hf_value"
    xml_file=$tag_name'_ZeroSuppressionLoader.xml'
    zip_file='./'$tag_name'_ZS.zip'
    zip $zip_file $xml_file
    rm $xml_file
    config_name='Zero Suppression data'
    uploadInstructions 'Zero suppression data' $zip_file
}

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
  echo ' 2. HTR Zero Suppression'
  echo ' 0. Contact info'
  
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
        zsMenu
	;;
      0)
      credits
      ;;
      *)
        echo 'Invalid choice - nothing to do...'
        credits
      ;;
  esac

}

if [ -e xmlToolsRun ]
then
    mainMenu
else
    compileMessage
fi
