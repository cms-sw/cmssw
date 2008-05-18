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
    echo -n 'Please compile the package. Issue gmake in'
    echo 'the current directory.'
    echo ''
}

uploadInstructions()
{
echo    ''
echo -n $1
echo -n ' are prepared for uploading to OMDS and saved in '
echo    $2
echo    ''
echo    'REMEMBER!'
echo -n 'It is always a good idea to upload to the validation '
echo    'database first before uploading to OMDS'
echo    ''
echo -n 'In order to upload to a database, copy '
echo -n $2
echo    ' to'
echo    'dbvalhcal@pcuscms34.cern.ch:conditions/ (validation - first!)'
echo    'dbprdhcal@pcuscms34.cern.ch:conditions/ (old master DB)'
echo    'dbpp5hcal@pcuscms34.cern.ch:conditions/ (old OMDS)'
echo    'your_user_name@srv-C2C02-06:/var/spool/xmlloader/hcal/prod/conditions/ (OMDS)'
echo    '(OMDS is on private .CMS network)'
echo    ''
echo -n 'You need to do all four in this order! If there is a problem '
echo -n 'reported from the validation DB, fix it first before '
echo -n 'proceeeding to Master and OMDS '
echo -n 'or, even better, follow the most recent instructions at '
echo    'https://twiki.cern.ch/twiki/bin/view/CMS/OnlineHCALDataSubmissionProceduresTOProdOMDSP5Server'
echo    ''
}


rbxPedMenu()
{
    echo ''
    echo '  -- Please choose the RBX config type'
    echo ' 1. Pedestals'
    echo ' 2. Zero delays'
    echo ' 3. GOL currents'
    echo ''
    echo -n 'Type: '
    read rbx_type_num

    case $rbx_type_num in
	1)
          rbx_type="pedestals"
          rbx_type_full="pedestals"
	  ;;
	2)
          rbx_type="delays"
          rbx_type_full="zero delays"
	  ;;
	3)
          rbx_type="gols"
          rbx_type_full="GOL currents"
	;;
	*)
          echo 'Invalid choice - nothing to do...'
          credits
	  exit 1
	  ;;
    esac

    echo    ''
    echo -n '  -- Processing RBX '
    echo $rbx_type_full
    echo    ''
    echo -n 'Please enter the path to the directory with '
    echo -n 'RBX pedestal XML brick files (read-only is fine)'
    echo    ''
    echo -n 'Path: '
    read ped_path_temp
    ped_path=`echo ${ped_path_temp%/}/`
    echo ''
    echo -n 'Please enter the desired tag name:'
    read tag_name
    echo ''
    echo -n 'Please enter the comment:'
    read comment
    echo ''
    echo -n 'Please enter the version (string):'
    read version
    echo 'processing RBX pedestals from' $ped_path '...'
    ped_files_num=`find $ped_path -iname "*.xml" | wc -l`
    echo $ped_files_num 'RBX pedestals brick files found...'
    if [ $ped_files_num -gt 0 ]
	then
	echo -n 'creating temp directory... '
	ped_temp_dir=`mktemp -d`
	echo $ped_temp_dir
	cp `find $ped_path -iname "*.xml"` $ped_temp_dir/

	#source rbx.sh $ped_temp_dir $tag_name

	ls $ped_temp_dir/*.xml > rbx_brick_files.list
	./xmlToolsRun --rbx=$rbx_type --filename=rbx_brick_files.list --tag=$tag_name --comment="$comment" --version="$version"
	zip -j ./$tag_name.zip $ped_temp_dir/*.oracle.xml
	rm rbx_brick_files.list
	echo -n 'Cleaning temporary files... '
	rm -rf $ped_temp_dir
      echo 'done'

    else
	echo 'No RBX pedestals brick files found... exiting'
    fi

    uploadInstructions 'RBX pedestals' './'$tag_name'.zip'
}


zsMenu()
{
    echo ''
    echo '  -- Processing HTR zero suppression config'
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
#      a_lut_file=`find $lut_temp_dir/ -iname "*_[0-9]*.xml" | awk 'NR==1{print $1}'`
      a_lut_file=`find $lut_temp_dir/ -iname "*_*[0-9].xml" | awk 'NR==1{print $1}'`
      echo 'DEBUG: ' $a_lut_file
      source luts.sh $a_lut_file
      echo -n 'Cleaning temporary files... '
      rm -rf $lut_temp_dir
      echo 'done'
  else
      echo 'LUT XML not found...check path'
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

# $1 type, 
genLutXml()
{
    echo ''
    echo -n 'Please enter the desired tag name:'
    read tag_name
    echo ''
    echo -n 'Please enter the comment:'
    read comment
    echo ''
    if [ $1 -eq 1 -o $1 -eq 3 ]
    then
	echo -n 'Linearization LUT master file:'
	read lin_master
	echo ''
    fi
    if [ $1 -eq 2 -o $1 -eq 3 ]
    then
	echo -n 'Compression LUT master file:'
	read comp_master
	echo ''
    fi
    echo -n 'Split XML files by crate? (y/n)'
    read split_by_crate
    echo ''
    
    ./xmlToolsRun --create-lut-xml --tag-name=$tag --lin-lut-master-file=$lin_master --comp-lut-master-file=$comp_master      
}

lutXml()
{
  echo ''
  echo '  -- LUT menu'
  echo ' 1. Generate linearization (input) LUT XML'
  echo ' 2. Generate compression (output) LUT XML'
  echo ' 3. Generate full set of LUT XML'
  echo ' 4. Prepare LUTs for uploading to the database'
  echo ' 0. Main menu'
  
  echo ''
  echo -n 'Please choose the action: '
  read line
  echo ''

  case $line in
      1)
        echo 'Generating linearization LUT XML...'
	echo ''
	_type=1
	;;
      2)
        echo 'Generating compression LUT XML...'
	echo ''
        _type=2
	;;
      3)
        echo 'Generating full set of LUT XML...'
	echo ''
        _type=3        
	;;
      4)
        lutMenu
	;;
      0)
      mainMenu
      ;;
      *)
        echo 'Invalid choice - nothing to do...'
        credits
      ;;
  esac
}

mainMenu()
{
  echo ''
  echo '  -- Main menu'
  echo ' 1. Trigger lookup tables'
  echo ' 2. HTR Zero Suppression'
  echo ' 3. RBX configuration'
  echo ' 0. Contact info'
  
  echo ''
  echo -n 'Please choose the configuration type: '
  read line
#  echo $line
  echo ''

  case $line in
      1)
        lutXml
	;;
      2)
        zsMenu
	;;
      3)
        rbxPedMenu
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
