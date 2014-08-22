validate_in_bash()
{
#vaildate_in_bash_begin_

local help='
performs validation of all functions in 
#$1 -- input file
in bash
'
if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi;

[[ -f some.sh ]] && rm some.sh;
cat <<_END >> some.sh  
Extractsomepart()
{
awk "BEGIN{ found=0 };  { if (index(\\\$0,\"#\") == 0 && index(\\\$0,\"\$2\")>0 &&  index(\\\$0,\"\$3\")>0 && index(\\\$0,\"\$2\")< index(\\\$0,\"\$3\")) {found=1}; if(found>0) print \\\$0; if (index(\\\$0,\"\$4\")==\$5)  found=0;}" \$1
}
_END

print_all_functions $1 | xargs -I {} echo "echo {} ; source  some.sh; Extractsomepart $1 {} \"()\" \"}\" 1 >| tmp.sh; source tmp.sh "| sh #

rm some.sh;
rm tmp.sh;

#vaildate_in_bash_end_
}

email_attachment() {
#email_attachment_begin_
local help='

send emails with attachment

#$1 -- to
#$2 -- cc
#$3 -- subject
#$4 -- body
#$5 -- file to be attached

Example:

email_attachment iggy.floyd.de@googlemail.com " " "Hi" "Hi"  21.03.12_higgsDesy.tgz
'

local title=`echo $@ | tr ' ' '_'`;


if [ $# -lt 5 ]; then print_help "$0" "$help"; return 1; fi;


    to="$1"
    cc="$2"
    subject="$3"
    body="$4"
    filename="${5:-''}"
    boundary="_====_blah_====_$(date +%Y%m%d%H%M%S)_====_"
    {

       print -- "To: $to"
        print -- "Cc: $cc"
        print -- "Subject: $subject"
        print -- "Content-Type: multipart/mixed; boundary=\"$boundary\""
        print -- "Mime-Version: 1.0"
        print -- ""
        print -- "This is a multi-part message in MIME format."
        print -- ""
        print -- "--$boundary"
        print -- "Content-Type: text/plain; charset=ISO-8859-1"
        print -- ""
        print -- "$body"
        print -- ""
        if [[ -n "$filename" && -f "$filename" && -r "$filename" ]]; then
            print -- "--$boundary"
            print -- "Content-Transfer-Encoding: base64"
            print -- "Content-Type: application/octet-stream; name=$filename"
            print -- "Content-Disposition: attachment; filename=$filename"
            print -- ""
            print -- "$(perl -MMIME::Base64 -e 'open F, shift; @lines=<F>; close F; print MIME::Base64::encode(join(q{}, @lines))' $filename)"
            print -- ""
        fi
        print -- "--${boundary}--"
    } | /usr/lib/sendmail -oi -t
#email_attachment_end_
}


prepare_presentation_area()
{
#prepare_presentation_area_begin_
local help='

create folder and file .odp for presentation

#$1 -- title of presentation to be put in file name

Example how to reduce the number of slides. 
It seems that there is a maximum around ~85 slides

cat example.rst| xargs -I {} echo " [[ \"{}\" = \"Figure\" ]] && a=\$((a+1)); [[ \$a -lt 82 ]] && echo \"{} \"; [[ \$a -ge 82  ]] &&  if [[  \"{}\" = \"Figure\" ||  \"{}\" = \"------\" ]]; then b=1 ;else echo \"{}\" ; fi ;  echo  " |sh
 
'

local title=`echo $@ | tr ' ' '_'`;


if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi;

 mkdir `presentation_name "higgsDesy"`;
 cd `presentation_name "higgsDesy"`;
 touch `presentation_name "marfin_$title.odp"`;
 cd -;
#prepare_presentation_area_end_
}

prepare_bruce_template()
{
#prepare_bruce_template_begin_
local help='

run program bruce to create presentation
'





echo ".. style::
   :layout.valign: center
   :align: center
   :font_size: 30

"

find `pwd` -iname "*jpg" -exec dirname {} \; | uniq | xargs -I {} echo ".. resource:: {}"

echo "

.. layout::
   bgcolor: silver
   quad:C#ffc0a0;V0,h;V0,h-88;Vw,h-88;Vw,h
   viewport:0,64,w,h-(64+48)


.. style::
        :layout.valign: center
        :align: left
        :font_size: 18
        :default.font_name: Times New Roma

"

#find `pwd` -iname "*jpg" -exec basename {} \; | uniq | xargs -I {} echo ".. |{}| image:: {}
#            :width: 400
#            :height: 400 
#
#"
find `pwd` -iname "*jpg"  | uniq | xargs -I {} echo " echo -n \".. |{}| image:: \";  basename {}; echo \"            :width: 400
            :height: 400 

\"  " |sh

echo ".. footer::
               I. Marfin TMVA input

"

# find `pwd` -iname "*jpg" -exec basename {} \; | uniq | xargs -I {} echo "Figure
#------
#
#- |{}|
#
#"

 find `pwd` -iname "*jpg"  | uniq | xargs -I {} echo "Figure
------

- |{}|

"

#prepare_bruce_template_end_
}


run_bruce()
{
#run_bruce_begin_
local help='

run program bruce to create presentation

#$1 -- file to be processed in full path

Here is an example of bruce presentation:

.. style::
   :layout.valign: center
   :align: center
   :font_size: 30

.. resource:: /usr1/scratch/marfin/presentations/
.. resource:: /usr1/scratch/marfin/presentations/21.03.12_higgsDesy/plots

.. layout::
   bgcolor: silver
   image:Desy-logo-small.gif;halign=right;valign=top
   quad:C#ffc0a0;V0,h;V0,h-88;Vw,h-88;Vw,h
   viewport:0,64,w,h-(64+48)


.. style::
        :layout.valign: center
        :align: left
        :font_size: 18
        :default.font_name: Times New Roman



.. |img1| image:: _can_Et2byEt1_0.jpg
        :width: 400
        :height: 400
.. |img2| image:: _can_Et3byEt1_0.jpg
        :width: 400
        :height: 400

.. footer::
        I. Marfin TMVA input

Collection of plots
------


Flavor content

- |img1|  |img2|
- |img1|  |img2|


'

if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi;


local bruce="/usr1/scratch/marfin/presentations/bruce-3.2.1/bruce.sh";
local output="`dirname $1`/bruce-output/`basename $1`";
cd `dirname $bruce`;
$bruce $1 --record=$output;
cd -;
 
#run_bruce_end_
}

test2()
{
#test2_begin_
local help='

NO HELP

FOR TEST PURPOSE
'

if [ "$1" = "help" ]; then print_help "$0" "$help"; return 1; fi;
if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi;

local filen=$1;

if [ "$1" = `basename $1` ]; then filen=`pwd`/$1;fi;


find `pwd` -type d -iname "*" -exec sh -c "echo \$1 ; cd \$1 ; $filen 2>/dev/null; cd - &> /dev/null;" _ {} \;

#local aaa="";
#local cmd="ls";
#local bbb="test2(){  ls -d */ | xargs -I {} echo \" echo {}; cd \`pwd\`/{}; $cmd; cd -; \" | sh }";
#bbb="test2(){  ls -d */ | xargs -I {} echo \" $bbb; echo {}; cd \`pwd\`/{}; $cmd; test2; cd -; \" | sh }";

#aaa="ls -d */ | xargs -I {} echo \"  echo \`pwd\`/{} ; cd \`pwd\`/{}; test2; cd -; \" "

#a="/usr1/scratch/marfin/presentations/utils.sh";

#echo $bbb

#ls -d */ | xargs -I {} echo "echo \`pwd\`/{} ; cd \`pwd\`/{};  cd -; " |sh;
#ls -d */ | xargs -I {} echo "source $a; echo \`pwd\`/{} ; cd \`pwd\`/{}; test2; cd -; " |sh
#ls -d */ | xargs -I {} echo "$bbb; echo \`pwd\`/{} ; cd \`pwd\`/{}; test2; cd -; " |sh
#test2_end_
}

apply_command()
{
#apply_command_begin_

local help='

apply command to all subfolders

#$1 -- command file

Example how to use it:

echo "rm *jpg"> mycmd; chmod a+x mycmd; apply_command mycmd; rm mycmd;
 echo "rm *jpg; rm *png">| mycmd; chmod a+x mycmd; apply_command mycmd; rm  mycmd;
'

if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi;


#local a;
#a=`find \`pwd\` -name $1`;
#[[ -n $a ]] &&  ls -d */ | xargs -I {} echo "echo {} ; cd {}; $a; cd - >& /dev/null" | sh


local filen=$1;
if [ "$1" = `basename $1` ]; then filen=`pwd`/$1;fi;

find `pwd` -type d -iname "*" -exec sh -c "echo \$1 ; cd \$1 ; $filen 2>/dev/null; cd - &> /dev/null;" _ {} \;


#apply_command_end_
}


presentation_name()
{
#presentation_name_begin_
local help="

get valid name for presentation folder 

"

if [ "$1" = "help" ]; then  print_help "$0" "$help"; return 1; fi;

local a;
a=${1+"$@"};


echo `date +"%d.%m.%y_$a"`;
#presentation_name_end_
}


dirs_sizes()
{
#dirs_sizes_begin_
local help="

print the usage of all sub-folders 
if you put option -h it will give you output in human-readable format

you can estimate total size in MB:

dirs_sizes  | awk 'BEGIN{aaa=0;} {aaa+=\$1;} END{print aaa/1024 \"M\"}'

"

if [ "$1" = "help" ]; then  print_help "$0" "$help"; return 1; fi;

ls -d */ | xargs -I {} find `pwd`/{} -type d -iname "*" | xargs -I {} echo " echo {} | du $1  " |sh
#dirs_sizes_end_
}

help() 
{ 
#help_begin_
local help='

print information message codded in the format
///usage: my message
///usage: here
etc

#$1 -- file with help 

'

if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi;

grep  --context=2  usage $1 |  sed  -ne 's/\/\/\/usage://p' 
#help_end_
}

update()
{
#update_begin_
local help='

copy recursively some file to sub-folders
etc

#$1 -- file 
'

   if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi


#ls -d */ | xargs -I {} find `pwd`/{} -name $1 | xargs -I {} echo " echo updating {}; bbb1=\` stat -c %Y $1 \`; bbb2=\` stat -c %Y {} \`; bbb3=\$(( \$bbb1 - \$bbb2 )); echo \" \$bbb1 \$bbb2 \$bbb3 \";  if [ \$bbb3 -lt 0 ]; then echo \"1\" >| aaa;  echo 222; else echo 111; fi;" | sh
#ls -d */ | xargs -I {} find `pwd`/{} -name $1 | xargs -I {} echo " echo -n \" updating {}     \"; bbb1=\` stat -c %Y $1 \`; bbb2=\` stat -c %Y {} \`; bbb3=\$(( \$bbb1 - \$bbb2 ));    if [ \$bbb3 -lt 0 ]; then echo \"1\" >| aaa;  echo \"   |  copying back   \";  cp {}  \`pwd\`/$1;  else echo \"   |  copying forward   \"; cp \`pwd\`/$1 {}; fi;" | sh
ls -d */ | xargs -I {} find `pwd`/{} -name $1 | xargs -I {} echo " printf \" %-100s \" \" updating {}     \"; bbb1=\` stat -c %Y $1 \`; bbb2=\` stat -c %Y {} \`; bbb3=\$(( \$bbb1 - \$bbb2 ));    if [ \$bbb3 -lt 0 ]; then echo \"1\" >| aaa;   printf \" %20s \" \"   |  copying back  \"; echo ;  cp {}  \`pwd\`/$1;  else printf \" %20s \" \"   |  copying forward  \"; echo ;  cp \`pwd\`/$1 {}; fi;" | sh

#[[ -f aaa ]] && echo aaa=`cat aaa`;

#[[ -f aaa ]] && [[ `cat aaa` -gt 0 ]] && { rm aaa; echo;echo; update $1 }
if [ -f aaa ]; then    rm aaa; echo;echo; update $1; fi;
[[ -f aaa ]] && rm aaa;
#update_end_
}

ListXMLParamters()
{
#ListXMLParamters_begin_
local help='

return list of the parameters corresponded to the xml tag in the file:
#$1 -- xml file
#$2 -- tag
'
        if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi


# sed -n  -e ':a' -e  "s/\(.*\)$1\(.*\)$1\(.*\)/\2;;;\1/p;ta" 
# sed -n -e ':a' -e 's/\(.*\)\(PDFDescriptor\)\(.*\)=\( *\)\([^ ]*\)/\2 \3/p;ta'  TMVAClassification_LikelihoodKDE.weights.xml | grep -v "="
# sed -n -e ':a' -e 's/\(.*\)\(PDF\)\( \{1,\}\)\(.*\)=\( *\)\([^ ]*\)/\2\3\4/p;ta'  TMVAClassification_LikelihoodKDE.weights.xml | grep -v "="
# sed -n -e ':a' -e 's/\"[^\"]*\"/aaa/g;s/\(.*\)\(PDF\)\( \{1,\}\)\(.*\)=\( *\)\([^ ]*\)/\2\3\4/p;ta'  TMVAClassification_LikelihoodKDE.weights.xml | grep -v "="

local file=$1
local tag=$2;

res=`echo " sed -n -e \':a\' -e \'s/\"[^\\\\\\"]*\"/aaa/g;s/\(.*\)\($tag\)\( \{1,\}\)\(.*\)=\( *\)\([^ ]*\)/\2\3\4/p;ta\' $file  | grep -v \"=\" | sed -ne \'s/\(.*\)\($tag\)\(.*\)/\1\3/p\' " | sh`;

echo "$res"

return 0;
#ListXMLParamters_end_
}

ReadXMLParameter()
{
#ReadXMLParameter_begin_
local help='

return parameter of the xml tag in the file:
#$1 -- xml file
#$2 -- tag 
#$3 -- parameter name
'
        if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi

local file=$1
local tag=$2;
local par=$3;

res=`echo " sed -ne 's/\(.*\)\($tag\)\(.*\)\($par\)\( *\)=\( *\)\([^ ]*\)\(.*\)/\7/p' $file " | sh`;

###remove all special symbols: ", >, <

echo "$res" | tr -d "\"" | tr -d ">" | tr -d "<";


return 0;

#ReadXMLParameter_end_
}



findLine()
{
#findLine_begin_
local help='

find a number of  line:
#$1 -- some word "A" in line
#$2 -- some word "B" in line
#$3 -- file name to be processed

cat $3 :

....
..... *.A.*B.*  --> line  will be printed
.....
.....
'
        if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi     

         sed -ne "/.*$1.*$2/p" $3
#findLine_end_
}

HowManyFieldsInFile()
{
#HowManyFieldsInFile_begin_

local help="
It tries to find the number of field in files having information to create objects of type:
	* runProof v2 (Sample)
	* runProof v3 (Sample)
	* Plotter
	* TMVA support
	* etc

Then file 'function.h' (or 'function_runProof.h', or similar) contains functions for reading files
	*Readinput_v2
	*Readinput_v3
	etc


#\$1 -- name of the file
"
 if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi

	res=`cat $1 | awk 'END{print NF}'`;
	echo "$res";

#HowManyFieldsInFile_end_
}

SplitListOfFiles()
{
#SplitListOfFiles_begin_
local help="

	If we have a list of root ntuples to be processed, then we could want to produce smaller lists of files to submit each of them to batch.
	It allows to increase a speed of calculations.
	The field corressponded  to number of files is being changed.

	Files called \$1_number will be produced

#\$1 -- name of the file containing
#\$2 -- how many files must be inside new lists

"

 if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi



local content="content";
local startpos=1;
local endpos=`echo "$startpos + $2 -1" | bc `;
local i=0;
while [  "$content" != "" ]
do
	i=$(($i+1));
	content=`cat $1 | sed -ne "${startpos},${endpos}p"`;

#	echo $startpos, $endpos
#	echo 222 "$content" 111;


	startpos=`expr $endpos + 1`;
	endpos=`echo "  $startpos + $2 -1"|bc`;
	local linesNum=`echo "$content" | wc -l`;
#	echo $linesNum;
	content=`echo "$content" | awk "{ if (NR==1 && \\$1 !=\"\") \\$6=$linesNum; print \\$0}"`

if [  "$content" != "" ]; then echo "$content" > $1_$i; fi;

done




return 0;

#SplitListOfFiles_end_
}


ListOfFilesForCrab()
{
local help="

	If we have a list of root ntuples to be processed, then we could want to produce smaller lists of files to submit each of them to crab

	Files called \$1_number will be produced

#\$1 -- name of the file containing
#\$2 -- job number
#\$3 -- how many jobs at all

"

 if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi


###Calculate delta: endpos = startpos + delta -1 
local numlines=`cat $1 | wc -l`;
local delta=`echo "$numlines/$3" | bc`;

if [ $delta -eq 0 ]; then delta=1; fi;

local content="content";
local startpos=`echo "($2-1)*$delta + 1" | bc`;
local endpos=`echo "$startpos + $delta -1" | bc `;
local limitpos=`echo "$delta*$3" | bc`;

content=`cat $1 | sed -ne "${startpos},${endpos}p"`;

if [ "$content" != "" ]
then
	 echo "$content" > $1_$2;
else
	echo " "> $1_$2;
	return 1;

fi

while [ "$endpos" -ge "$limitpos" -a  "$content" != "" ]
do

#	echo $startpos, $endpos
#	echo 222 "$content" 111;


	startpos=`expr $endpos + 1`;
	endpos=`echo "  $startpos + $delta -1"|bc`;

	content=`cat $1 | sed -ne "${startpos},${endpos}p"`;

#	local linesNum=`echo "$content" | wc -l`;
#	echo $linesNum;
#	content=`echo "$content" | awk "{ if (NR==1 && \\$1 !=\"\")  print \\$0}"`

if [  "$content" != "" ]; then echo "$content" >> $1_$2; fi;

done


return 0;

#_end_
}



createCPPCode()
{

local help="
convert shell function to ROOT macros
#\$1 -- name of the function
"
 if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi

args=`$1 | sed -ne 's/.*#$\(.*\).*/\1/p' | awk '{print $1}'`


if [ -f $1.C ]; then rm $1.C; fi;

touch $1.C;


argstr="";

#echo $args

##Forming call line of function
for i in $args
do
argstr=`echo "${argstr}TString arg$i,"`;
done

argstr=`echo "${argstr}TString pathArg=\"./\","`;
argstr=`echo "${argstr}Bool_t verbose=kFALSE,"`;


argstr=`echo "(${argstr%%,})"`;


headline=`echo "TString $1$argstr{"`;
#codeline1=($(echo "TString cmd=TString(\"$1 $argstr2 \");"));
#sections=($(FindSection "[" "]" "$1"));
unset codeline;
codeline[$((${#codeline[@]}+1))]=`echo "TString cmd=TString(\"source \");"`;
codeline[$((${#codeline[@]}+1))]=`echo " cmd+=pathArg+TString(\"utils.sh; \");"`;
codeline[$((${#codeline[@]}+1))]=`echo "cmd+=TString(\"$1 \" );"`;

for i in $args
do
codeline[$((${#codeline[@]}+1))]=`echo "cmd+=TString(' ')+arg$i+TString(' ');"`;
done



#codeline[$((${#codeline[@]}+1))]=`echo "FILE * myPipe = gSystem->OpenPipe(cmd.Data(),\"r\");"`;
codeline[$((${#codeline[@]}+1))]=`echo "TString res =gSystem->GetFromPipe(cmd.Data()) ;"`;
#codeline[$((${#codeline[@]}+1))]=`echo "res.Gets(myPipe);"`;
#codeline[$((${#codeline[@]}+1))]=`echo "gSystem->ClosePipe(myPipe);"`;




printline=`echo "if (verbose) std::cout<<res<<std::endl;"`;
returnline=`echo "return res;}"`;

###Printing to file
output="///run me"
echo $output >> $1.C;
output="///  $1.C$argstr";
echo $output >> $1.C;


echo >>$1.C
echo >>$1.C

echo "#include \"TString.h\"" >>$1.C
echo "#include \"TSystem.h\"" >>$1.C
echo "#include <iostream>" >>$1.C

echo >>$1.C
echo >>$1.C



echo $headline >>$1.C
for ((i=1;i<=${#codeline[@]};i++))
do
echo ${codeline[$i]} >>$1.C;
done
echo $printline >>$1.C
echo $returnline >>$1.C;


cat $1.C

return 0;

#_end_
}

FindSection()
{

local help="

find a text from one pos to another pos
It's useful to process multicrab.cfg having format :

[section1]

class1.atribute1=val11
class1.atribute2=val12
class1.atribute3=val13



[section2]

class2.atribute1=val21
class2.atribute2=val22
class2.atribute3=val23

etc


#\$1 -- pos1
#\$2 -- pos2
#\$3 -- file to  be processed
"

        if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi

local pos1=$1;
local pos2=$2;
local file=$3;


cat $file | sed "/#/d" |  awk -v pos1=$pos1 -v pos2=$pos2  "BEGIN {c=0} { if( index(\$$0,pos1)>0) {c=NR } if(c>0) { print \$$0}  if (index(\$$0,pos2) && c>0)  {c=0} }" | tr -d "[" | tr -d "]" |  tr -d "#";


return 0;
}

ProcessMulticrab_cfg()
{
local help="


find all sections and fill assoc array in the format:
arr["section1_class1.atribute1"]=val11;
arr["section2_class2.atribute1"]=val21;
etc

Then *_job_config files are produced which might be used to run ProofCluster in the batch job

assocArray.sh must be in the same dir as utils.sh

#\$1 -- file to be processed, like multicrab.cfg
"

if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi

### Let's find assoc array support

removeAllElement
cc=`findElement 2>&1`;

b=`echo "$cc" |  awk '{if (index($0, command not found)>0) {print 1 }}'`;


if [ "$b" -eq 1 -a  -f  "assocArray.sh" ]
 then 
	echo "No assoc array support found. Try to ini it";
	source ./assocArray.sh;
	a=`findElement 2>&1`;
	else
	echo "No assocArray.sh. return";
	return 1;
fi




##Fill temporary array with section's names
sections=($(FindSection "[" "]" "$1"));
len=${#sections[@]};
#echo $len
 for ((j=1;j<=$len;j++))
                do
#		echo j=$j		
		beg=`findLineNumber "${sections[$j]}" "]" $1 |   tr " " "\n" | tail -1`;
		end=`findLineNumber "${sections[$((j+1))]}" "]" $1  | tr " " "\n" | tail -1`;

		echo "${sections[$((j))]}"
		echo "${sections[$((j+1))]}"
	
#		echo "beg=$beg";
#		echo "end=$end";

		if [ "$beg" = "$end" ]; then end=`cat $1 | wc -l`; fi;


		curtext=`sed -ne "$((beg+1)),$((end-1))p" < \`echo $1\``;


#removing comments with '#'
		 curtext=`echo $curtext |  sed "s/\ *#.*\ *//g"`;

#removing white space araound '='	
		curtext=`echo $curtext | sed  "s/ *= */=/g"`;
#		echo 11${curtext}22;

##Fill array!
		if [ -n "$curtext"  -a -n "${sections[$j]}" ]		
		then
			curtext2=($(echo "$curtext" | tr ' ' '\n' ));
			len2=${#curtext2[@]};
			for (( k=1;k<=$len2;k++ ))
			do
				elem=`echo "${curtext2[$k]}" | sed -ne "s/\(.*\)=\(.*\)/\1/p"`;
				val=`echo "${curtext2[$k]}" | sed -ne "s/\(.*\)=\(.*\)/\2/p"`;
				addElement "${sections[$j]}_$elem"  $val;
#				echo "adding ${sections[$j]}";
			done
		
		fi


		done




###List of available keywords
#USER.publish_data_name 
#CMSSW.datasetpath
#CMSSW.pset 
#CMSSW.total_number_of_events
#CMSSW.dbs_url 
##USER.type=SGN,BKG or DATA  # my new keywords!
##USER.weight=0.0025 #
#findElement "SingleTopBar_s_Trees_CMSSW.dbs_url"


###Produce pathes to files!
which voms-proxy-init >& /dev/null;
found=$?;
if [ "$found" -eq 1 ] ; then "ini GLITE32!"; return 1; fi;

len=${#sections[@]};
 for ((j=1;j<=$len;j++))
                do
		datasetpath=`findElement "${sections[$j]}_CMSSW.datasetpath"`;
		dbs_url=`findElement "${sections[$j]}_CMSSW.dbs_url"`;

		if [ -z "$dbs_url" ]; then  dbs_url="http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet";fi;
		
		if [ -n "$datasetpath" -a -n "$dbs_url" ]
		then

		echo files for ${sections[$j]}

##Find files
	files=`echo	"dbs search --url=\"$dbs_url\" --query=\"find file where dataset=$datasetpath\" | sed \"1,4d\""  | sh`;
		addElement "${sections[$j]}_files" "$files"
	       # files=($(echo "$files" | tr ' ' '\n' ));
	#	echo "${files[0]}"
		fi

		done



#final config_job file
if [ -f config_job ]; then rm job_config; fi;
touch job_config;

len=${#sections[@]};
for ((j=1;j<=$len;j++))
	do

		files=`findElement "${sections[$j]}_files"`;
		type=`findElement "${sections[$j]}_USER.type"`;
		weight=`findElement "${sections[$j]}_USER.weight"`;
		name=${sections[$j]};

		if [ -z "$name" ] ; then continue; fi;
		if [ -f ${name}_job_config ]; then rm ${name}_job_config; fi;
		touch ${name}_job_config;


		files=($(echo "$files" | tr ' ' '\n' ));
		files_len=${#files[@]};
		for ((k=1;k<=${files_len};k++))
		do
		output=`echo "dcap://dcache-cms-dcap.desy.de:22125//pnfs/desy.de/cms/tier2${files[$k]} $name $type"`;
#		echo "$output"
		if [ "$type" = "DATA" ]; then output=`echo "$output config_data.py" `; else output=`echo "$output config_mc.py"`; fi
		output=`echo "$output $weight"`;
		 if [ $k -eq 1 ]; then  output=`echo "$output $files_len"`; else output=`echo "$output 0"`; fi
		echo "$output" >> job_config;
		echo "$output" >> ${name}_job_config	
		done
	done



#findElement "SingleTopBar_s_Trees_USER.type"
#findElement "SingleTopBar_s_Trees_USER.weight"

return 0;
#_end_
}

ProcessMulticrab_cfg_v2()
{
#ProcessMulticrab_cfg_v2_begin_
local help="


find all sections and fill assoc array in the format:
arr["section1_class1.atribute1"]=val11;
arr["section2_class2.atribute1"]=val21;
etc

Then *_job_config files are produced which might be used to run ProofCluster in the batch job

assocArray.sh must be in the same dir as utils.sh

Uses das_client.py. This file must be accesiable

#\$1 -- file to be processed, like multicrab.cfg

"

if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi

### Let's find assoc array support

removeAllElement
cc=`findElement 2>&1`;

b=`echo "$cc" |  awk '{if (index($0, command not found)>0) {print 1 }}'`;


if [ "$b" -eq 1 -a  -f  "assocArray.sh" ]
 then 
	echo "No assoc array support found. Try to ini it";
	source ./assocArray.sh;
	a=`findElement 2>&1`;
	else
	echo "No assocArray.sh. return";
	return 1;
fi

if [ ! -f "das_client.py" ]; then echo "No das_client.py found, return";   return 1; fi;



##Fill temporary array with section's names
sections=($(FindSection "[" "]" "$1"));
len=${#sections[@]};
#echo $len
 for ((j=1;j<=$len;j++))
                do
#		echo j=$j		
		beg=`findLineNumber "${sections[$j]}" "]" $1 |   tr " " "\n" | tail -1`;
		end=`findLineNumber "${sections[$((j+1))]}" "]" $1  | tr " " "\n" | tail -1`;

		echo "${sections[$((j))]}"
		echo "${sections[$((j+1))]}"
	
#		echo "beg=$beg";
#		echo "end=$end";

		if [ "$beg" = "$end" ]; then end=`cat $1 | wc -l`; fi;


		curtext=`sed -ne "$((beg+1)),$((end-1))p" < \`echo $1\``;


#removing comments with '#'
		 curtext=`echo $curtext |  sed "s/\ *#.*\ *//g"`;

#removing white space araound '='	
		curtext=`echo $curtext | sed  "s/ *= */=/g"`;
#		echo 11${curtext}22;

##Fill array!
		if [ -n "$curtext"  -a -n "${sections[$j]}" ]		
		then
			curtext2=($(echo "$curtext" | tr ' ' '\n' ));
			len2=${#curtext2[@]};
			for (( k=1;k<=$len2;k++ ))
			do
				elem=`echo "${curtext2[$k]}" | sed -ne "s/\(.*\)=\(.*\)/\1/p"`;
				val=`echo "${curtext2[$k]}" | sed -ne "s/\(.*\)=\(.*\)/\2/p"`;
				addElement "${sections[$j]}_$elem"  $val;
#				echo "adding ${sections[$j]}";
			done
		
		fi


		done




###List of available keywords
#USER.publish_data_name 
#CMSSW.datasetpath
#CMSSW.pset 
#CMSSW.total_number_of_events
#CMSSW.dbs_url 
##USER.type=SGN,BKG or DATA  # my new keywords!
##USER.weight=0.0025 #
#findElement "SingleTopBar_s_Trees_CMSSW.dbs_url"


###Produce pathes to files!
#which voms-proxy-init >& /dev/null;
#found=$?;
#if [ "$found" -eq 1 ] ; then "ini GLITE32!"; return 1; fi;

len=${#sections[@]};
 for ((j=1;j<=$len;j++))
                do
		datasetpath=`findElement "${sections[$j]}_CMSSW.datasetpath"`;
		dbs_url=`findElement "${sections[$j]}_CMSSW.dbs_url"`;

		if [ -z "$dbs_url" ]; then  dbs_url="http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet";fi;
		
		if [ -n "$datasetpath" -a -n "$dbs_url" ]
		then

		echo files for ${sections[$j]}

##Find files
	files=` ./das_client.py --query="file dataset=/2B2C1Jet_TuneZ2_7TeV-alpgen-pythia6/Summer11-PU_S4_START42_V11-v1/AODSIM | grep file.name" --format=plain --idx=0 --limit=100000 | awk '{if(NR>3) print \$0}'`
		addElement "${sections[$j]}_files" "$files"
	        # files=($(echo "$files" | tr ' ' '\n' ));
		# echo "${files[0]}"
		fi

		done



#final config_job file
if [ -f config_job ]; then rm job_config; fi;
touch job_config;

len=${#sections[@]};
for ((j=1;j<=$len;j++))
	do

		files=`findElement "${sections[$j]}_files"`;
		type=`findElement "${sections[$j]}_USER.type"`;
		weight=`findElement "${sections[$j]}_USER.weight"`;
		name=${sections[$j]};

		if [ -z "$name" ] ; then continue; fi;
		if [ -f ${name}_job_config ]; then rm ${name}_job_config; fi;
		touch ${name}_job_config;


		files=($(echo "$files" | tr ' ' '\n' ));
		files_len=${#files[@]};
		for ((k=1;k<=${files_len};k++))
		do
		output=`echo "dcap://dcache-cms-dcap.desy.de:22125//pnfs/desy.de/cms/tier2${files[$k]} $name $type"`;
		echo "$output"
		if [ "$type" = "DATA" ]; then output=`echo "$output config_data.py" `; else output=`echo "$output config_mc.py"`; fi
		output=`echo "$output $weight"`;
		 if [ $k -eq 1 ]; then  output=`echo "$output $files_len"`; else output=`echo "$output 0"`; fi
		echo "$output" >> job_config;
		echo "$output" >> ${name}_job_config	
		done
	done



#findElement "SingleTopBar_s_Trees_USER.type"
#findElement "SingleTopBar_s_Trees_USER.weight"

return 0;
#ProcessMulticrab_cfg_v2_end_
}



findLineNumber()
{
#findLineNumber_begin_

local help='

find a number of  line:
#$1 -- some word "A" in line
#$2 -- some word "B" in line
#$3 -- file name to be processed

cat $3 :

....
..... *.A.*B.*  --> line # will be printed
.....
.....

'

	if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi	
         sed -ne "/.*$1.*$2/=" $3

#findLineNumber_end_
}


print_all_functions()
{
#print_all_functions_begin_
local help='
print all functions in the file 

#$1
'

      if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi

cat $1 |  sed -ne 's/\(.*\)()$/\1/p' | sed '/#/d' | sed '/#$/d';

#print_all_functions_end_
}

print_all_functions_help()
{
#print_all_functions_help_begin_
local help='
print help of all functions in the file

#$1
'

      if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi

a=`print_all_functions $1`;
a=($(echo $a | tr ' ' '\n'));
for ((i=1;i<=${#a[@]};i++))
do
	${a[$i]};

done

#print_all_functions_help_end_
}


field()
{
#field_begin_	
local help='
from string 
#$1 
get field 
#$2 
'


       if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi

         echo "$1" | awk  "{print \$$2 }"

#field_end_
}

print_help()
{
#print_help_begin_
if [ $# -lt 2 ]; then return 1; fi

echo ""
echo "$1"
echo "usage:"
echo ""
echo "$2";

#print_help_end_
}

PowerToExp()
{
#PowerToExp_begin_

local help='
make transformation of the form from  0.012 to 1.2e-02
Current format is:
	4 -- number fields in mantisa

#$1 -- input number
'
        if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi


 echo "$1" | awk  '{printf "%5.4e\n",$0}'


 

return 0;

#PowerToExp_end_
}

ExpToPower()
{

#ExpToPower_begin_
local help='

make transformation of the form from 1.2e-2 to 0.012
#$1 -- input number
'
        if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi


local res="$1";

local ind=`expr index "$1" "e"`;
if [ $ind -gt 0 ]
then

local mantisa=`echo "$1" |  sed -e 's/\(.*\)e\(.*\)/\1/'`
 res="$mantisa";
local exponenta=`echo "$1" |  sed -e 's/\(.*\)e\(.*\)/\2/'`

exponenta=`echo $exponenta | sed -e 's/+0*\(.*\)/\1/'` 

if [ -z $exponenta ]; then exponenta=0; fi;

if [ $exponenta -gt 0 ]
then
        for (( i = 0; i < $exponenta; i++ ))
        do
                res=`echo "scale=5; $res * 10" | bc `;
        done
else
         for (( i = 0; i > $exponenta; i-- ))
        do
                res=`echo "scale=20; $res / 10" | bc `;
        done
fi
fi
echo $res
return;

#ExpToPower_end_
}

	
Change_word_comment_line()
{
#Change_word_comment_line_begin_
local help='
change the word in line; comment old line; insert a new line with changed "word"

Change_word_comment_line:

#$1 needed to find line
#$2 needed to find line
#$3 word to replace by
#$4 new word
#$5 comment symbol
#$6 the file to change
#$7 the new file
'


if [ $# -lt 7 ]; then print_help "$0" "$help"; return 1; fi

#if $4 contains '/'
local newString=`echo  $4 | sed 's%/%\\\\/%g'`


	 change_line=`findLine "$1" "$2" "$6"`;
         numline=`findLineNumber "$1" "$2" "$6" `
	where_to_insert=`echo "scale=0; $numline+1"|bc`


#echo $change_line
#echo $numline
#echo $where_to_insert

#continue;

	if [ -n "${change_line}" ]
	then



		change_newline1=`echo "${change_line}" | sed -e "s/$3/$newString/"`;
		change_newline2=`echo "$5${change_line}"`;

#			echo $change_newline1
#			echo $change_newline2
		
		fl_tmp=tmp_$$;
		fl_tmp2=tmp2_$$;

		sed -e "
		${numline}{
		c\
		 $change_newline2\n
		}" $6 >  ${fl_tmp}; 
#		mv  ${fl_tmp}  $7;


		sed -e "
                ${where_to_insert}{
                c\
                 $change_newline1
                }"   ${fl_tmp} >  ${fl_tmp2}; 
                mv  ${fl_tmp2}  $7;


	rm ${fl_tmp};

	fi

#Change_word_comment_line_end_
}

Make_Table()
{
#Make_Table_begin_
local help='
#$1 header of table
#
#$2 file containing table in groff(tbl) format:
#
#	aa ;	bb ;	cc
#	dd ;	vv ;	etc
#
#
#
#$3 -- type: ps or ascii
'

if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi


#Define format! --> Maximal number of fields!

#Initial max of NF:
maxNF=`cat $2 | sed -n "1p" | awk "{print NF}"`;
nl=`cat $2| wc -l`;


for (( i=1; i<=$nl;i++ ))
do

curLine=`cat $2 | sed -ne "${i}p"`;


curNF=`echo $curLine |  awk -F';' '{print NF}'`;
cmpr=`echo "$maxNF > $curNF" | bc`;

if [ $cmpr -eq 0 ] ; then maxNF=$curNF; fi 

done



echo '.TS
allbox tab(;);' > tmp.file;
for (( i=1; i<=$maxNF;i++ ))
do
echo -n "c" >> tmp.file;
done
echo  '.' >> tmp.file;
cat $2 >> tmp.file;
echo '.TE' >> tmp.file;


	echo "		$1	";

# tbl table.tbl | troff -Tascii | grotty 2>|/dev/null  | sed -e '/^$/d'
if [  "$3" = 'ascii' ];
then
tbl tmp.file | troff -Tascii |  grotty 2>|/dev/null  | sed -e '/^$/d';
fi;

if [  "$3" = 'ps' ];
then
tbl tmp.file | troff -Tps |  grops 2>|/dev/null > $2.ps;
fi;
rm tmp.file;
return 0;
#Make_Table_end_
}

Find_vars_to_replace()
{
#Find_vars_to_replace_begin_
local help='
#$1 vars definitions: like %%anyVar%% -->%%
#$2 files to process: string type!!
'


if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi

#find the number of input files:
len=`echo $2 | tr " " ":"  | awk -F":" ' END{ print NF}'`;


	echo ">>   `pwd`";

textFileNames=();
textSettings=();
textCfg=();


for (( i=1;i<=$len;i++ ))
do
	curFileName=`echo $2  | awk  "{print \\$${i}}" `;
	lineNumber=`sed -ne "/.*$1.*$1.*/=" $curFileName`;

		echo ">>   $curFileName   >> contains >> ";
		echo ;
		echo ;


	for k in $lineNumber
	do

	curLine=`sed -ne "${k}p" $curFileName`;
#	echo curLine $curLine 

	if [ -n "$curLine" ] 
	then 

	textFileNames=($(echo $textFileNames | tr ' ' '\n') $curFileName);

	tempVars=`echo $curLine | sed -n  -e ':a' -e  "s/\(.*\)$1\(.*\)$1\(.*\)/\2;;;\1/p;ta"  >| tmp.file; sed -ne "\`cat tmp.file | wc -l\`p" tmp.file; rm tmp.file`;
#	echo "tempvar1" $tempVars;
	tempVars=`echo $tempVars | sed -ne 's/\(.*\)\(;;;.*$\)/\1/p' | tr ';;;' ' '`;
#	echo "tempvar2"  $tempVars;

		vars=($( echo "$tempVars" | tr ' ' '\n'));
		

#		echo "${#vars[*]}"
#		echo "vars3" ${vars[0]};
#		echo "vars3" ${vars[1]};
#		echo "vars3" ${vars[2]};
#		echo "vars3" ${vars[3]};
#
#continue;

		for ((j=0;j<${#vars[@]};j++))
		do

		echo "export ${vars[$j]}=?  		# >> code for 'settings.cfg' >> "
	

		done


		for ((j=0;j<${#vars[@]};j++))
		do

		echo "Changeword '$1${vars[$j]}$1' \$${vars[$j]} $curFileName $curFileName;     # >> code for 'ini.cfg' >> "


		done

	
	fi



	done	

	

done

return 0;
#Find_vars_to_replace_end_
}

Changefield()
{
#Changefield_begin_
local help='
#$1 needed to find line
#$2 needed to find line
#$3 number of field to replace
#$4 new value
#$5 the file to change
#$6 the new file 
'


if [ $# -lt 6 ]; then print_help "$0" "$help"; return 1; fi


	
	 change_line=`findLine "$1" "$2" "$5"`;
	 numline=`findLineNumber "$1" "$2" "$5" `


	if [ -n "${change_line}" ]
	then

		 change_field=`echo "${change_line}" | awk  "{print \\$$3 }"`; 
		 change_newline=`echo "${change_line}" | sed -e "s/${change_field}/$4/"`;
		fl_tmp=tmp_$$;

		sed -e "
		${numline}{
		c\
		 $change_newline
		}" $5 >  ${fl_tmp}; 
		mv  ${fl_tmp}  $6;


	fi
#Changefield_end_	
}


Changeword()
{
#Changeword_begin_
local help='
#$1 old word to be replaced by
#$2 new word
#$3 old file
#$4 new file
'


if [ $# -lt 4 ]; then print_help "$0" "$help"; return 1; fi



local tmpfile=$$_tmp_1;

#if $1 contains '/'
local newString=`echo  $2 | sed 's%/%\\\\/%g'` 

#echo $newString

cat $3 | sed "s/$1/$newString/" > $tmpfile; 

mv $tmpfile $4;
#Changeword_end_
}

Random_number()
{
#Random_number_begin_
local ISEED=`dd if=/dev/urandom count=1 2> /dev/null | cksum | cut -b-5`
echo $ISEED
#Random_number_end_
}

Combine_content()
{
#Combine_content_begin_
local help='
###produces the list :  file1,
##			file2,
##			file3	
##			etc
#$1 input file
'


if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi


local total_lines=`cat $1 | wc -l`
local content="";
for ((i=1;i<=$total_lines;++i))
do	

	tmp_read_line=`sed -ne ${i}p  $1`


	if [ -n "${tmp_read_line}" -a  $i -lt $total_lines ]
		then
			content="${content} \n
				 \"${tmp_read_line}\","
		fi

	if [ -n "${tmp_read_line}" -a  $i -eq $total_lines ]
		then
			content="${content} \n
				 \"${tmp_read_line}\""
		fi

done
		
	
		echo -e $content
#Combine_content_end_
}




Insert_here()
{
#Insert_here_begin_
local help='
Insert_here:

#$1 -- file to be modified
#$2 -- where to insert
#$3 -- file to be inserted
'


if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi

	if [ -n $2 -a -n $1 -a -n $3 ]
	then 
		local tmp_file="$$_file_tmp";
		local tmp_file2="$$_file_tmp2";
		cp $1 $tmp_file;
		sed -e "/$2/r $3" < $tmp_file > $tmp_file2;
		rm $tmp_file;
		mv $tmp_file2 $1
	else 
		
		echo "Insert_here():: Something is wrong"
	fi
	
#Insert_here_end_
}

FixAlpgenLHE()
{
#FixAlpgenLHE_begin_

local help='
#$1 -- file to be modified
#$2 -- from to be removed
#$3 -- until to be removed
'

if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi

if [ -n $2 -a -n $1 -a -n $3 ]
        then
                local tmp_file="$$_file_tmp";
                local tmp_file2="$$_file_tmp2";
                cp $1 $tmp_file;
###Print only first first field of the current string match to \$2
awk "BEGIN{act=0} { if (match(\$0,\"$3\")) act=0; if (act==0 && !match(\$0,\"$2\") )  print \$0; if (act>0) { }; if (match(\$0,\"$2\")){ act=1; print \$1}; }" $tmp_file > $tmp_file2;
	
                mv $tmp_file2 $1;
                rm $tmp_file;
fi

#FixAlpgenLHE_end_
}


ValidateAlpgenLHE()
{
#ValidateAlpgenLHE_begin_
local help='
#$1 -- file to be validated
#$2 -- from to be validated
#$3 -- until to be validated
'

if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi

if [ -n $2 -a -n $1 -a -n $3 ]
        then
                local tmp_file="$$_file_tmp";
                local tmp_file2="$$_file_tmp2";
                cp $1 $tmp_file;
###Print only first first field of the current string match to \$2
awk "BEGIN{act=0} { if (match(\$0,\"$3\")) {act=0; print \$0 ; }; if (act==0 ) { }; if (act>0) { prin \$0 }; if (match(\$0,\"$2\")){ act=1 ; print \$0}; }" $tmp_file > $tmp_file2;
	
                cat $tmp_file2 
                rm $tmp_file2;
                rm $tmp_file;
fi

#ValidateAlpgenLHE_end_
}

TestFunction()
{
#TestFunction_begin_

local help='
#$1 -- file to be processed
#$2 -- name of function to be tested

return 0 if all is ok.
return 1 if something is bad.
'
if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi

rm  $$_fsdsadf_tmp >& /dev/null;
ExtractFunctionBody  $1 $2 > $$_fsdsadf_tmp;

#cat $$_fsdsadf_tmp; ;
source $$_fsdsadf_tmp;
echo "Try to get help, running the command without parameters"
($2) | grep "usage" >& /dev/null;echo $?
echo "Try to get help, running the command with help parameter"
($2 "help") | grep "usage" >& /dev/null;echo $?  
rm  $$_fsdsadf_tmp;

#TestFunction_end_
}

HLT_TREE()
{
#HLT_TREE_begin_


local help=' 

it ceates HLT structure:

Example
we want to view the structure of all sequences in the path  
HLT_Jet60Eta1p7_Jet53Eta1p7_DiBTagIP3DFastPV_v1

where sequences matching patterns "hltL1sL1.*" "HLTBeg.*" "hltPre.*" "HLTDo.*" "HLTEnd.*" "HLT.*Corr.*" are excluded 

HLT_TREE HLT_GRun_cff.py HLT_Jet60Eta1p7_Jet53Eta1p7_DiBTagIP3DFastPV_v1 " "  "hltL1sL1.*" "HLTBeg.*" "hltPre.*" "HLTDo.*" "HLTEnd.*" "HLT.*Corr.*"


Notation: 

1) cms.Path
+++ Name +++ 

2) cms.Sequence
--- Name ---

3) cms.EDFilter or cms.Producer module`
*** Name ***

4) input tags of a module
~~~ Name ~~~
'

if [ $# -lt 3  ]; then print_help "$0" "$help"; return 1; fi
if [ "$1" = help ];  then print_help "$0" "$help"; return 1; fi

local irrelevant_modules;

local a1;
local a2;
local a3;

local args;
args=("$@");
local add;
for ((i=4;i<=$#;i++))
do
        add=${args[$i]};
        irrelevant_modules=`echo $irrelevant_modules $add`;

done

a1=($( HLT_PATH $1 $2 $irrelevant_modules ));
a2=($( HLT_SEQUENCE $1 $2 $irrelevant_modules ));
a3=($( HLT_MODULE $1 $2 $irrelevant_modules ))

#echo "a1 = $a1"
#echo "a2 = $a2"
#echo "a3 = $a3"



local b1;
local b2;
local c;
b1=0;
b2=0;

if [ -n "$a1" ]; then echo "$3 +++ $2 +++";b1=1;fi;
if [ -n "$a2" ]; then echo "$3 --- $2 ---";b2=1;fi;
if [ -z "$a1" -a -z "$a2" ]; then echo "$3 *** $2 ***";fi;

if [ -n "$a3" ]
then 
	for j in "${a3[@]}"
	do
		echo "$3	~~~ $j ~~~";
	done
fi
#for (( i=1;i<=${#a[@]};i++ ))
if [ $b1 -gt 0 ]
then
	for j in "${a1[@]}"
	do
#	echo HLTPATH $j
#	bbb=($( HLT_SEQUENCE $1 $j  $irrelevant_modules ));
	HLT_TREE $1 $j  "$3	" $irrelevant_modules
	done
fi


if [ $b2 -gt 0 ]
then
	for j in "${a2[@]}"
	do
#	echo HLTSEQ $j
#	bbb=($( HLT_SEQUENCE $1 $j  $irrelevant_modules ));
	HLT_TREE $1 $j "$3	" $irrelevant_modules
	done
fi


#HLT_TREE_end_
}

HLT_PATH()
{
#HLT_PATH_begin_	
local help='

That is the part of processing HLT_MENU.cff file

#$1 -- HLT MENU file
#$2 -- HLT path , exact name
#$3 -- what sequences and modules to ignore

return the list of modules and HLT sequences
'

if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi

local a;
#a=($(grep $2 $1 | grep "cms.Path" | tr ' ' '\n'))
a=($(grep -w -E "^$2"  $1 | grep "cms.Path" ))
#echo "$a"

local irrelevant_modules;
irrelevant_modules="$2";

local args;
args=("$@");
local add;
for ((i=3;i<=$#;i++))
do
        add=${args[$i]};
        irrelevant_modules=`echo $irrelevant_modules $add`; 

done

 
#echo 1212232 $irrelevant_modules
#echo 1212232 
b=$(Remove_parts_string "$a" "cms.Path" "(" ")"  "+" "=" $irrelevant_modules | tr ' ' '\n')
echo "$b"

#HLT_PATH_end_
}

HLT_SEQUENCE()
{
#HLT_SEQUENCE_begin_	
local help='

That is the part of processing HLT_MENU.cff file

#$1 -- HLT MENU file
#$2 -- HLT SEQUENCE , exact name
#$3 -- what sequences and modules to ignore
return the list of modules in HLT sequences

Example 
 HLT_SEQUENCE HLT_GRun_cff.py  HLTBTagIPSequenceL25bbPhiL1FastJet "HLTDoLocalPixelSequence hltSelector4JetsL1FastJet"

'

if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi

local a;
#a=($(grep $2 $1 | grep "cms.Path" | tr ' ' '\n'))
a=($(grep -w -E "^$2" $1 | grep "cms.Sequence" ))
#a=`grep -w $2 $1 | grep "cms.Sequence" `

#echo $a
#echo 
#echo 
#echo

local b;

local irrelevant_modules;
irrelevant_modules="$2";

local args;
args=("$@");
local add;
for ((i=3;i<=$#;i++))
do
        add=${args[$i]};
        irrelevant_modules=`echo $irrelevant_modules $add`;

done


#Remove_parts_string "$a" "cms.Sequence" "(" ")" "+" "="  $irrelevant_modules 
b=$(Remove_parts_string "$a" "cms.Sequence" "(" ")" "+" "="  $irrelevant_modules | tr ' ' '\n')
echo "$b"


#HLT_SEQUENCE_end_
}


HLT_MODULE()
{
#HLT_MODULE_begin_
local help='

That is the part of processing HLT_MENU.cff file

#$1 -- HLT MENU file
#$2 -- HLT MODULE , exact name
#$3 -- what input tags to ignore
return the list of input tags  of the modules 

Example
	no example
'

if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi
local a;


a=($(Extract_some_part $1 $2 "=" ")" 1 | grep -i "inputtag" | awk '{print $4}' | tr ' ' '\n'))
#a=`grep -w $2 $1 | grep "cms.Sequence" `

#echo "$a"


local irrelevant_modules;

local args;
args=("$@");
local add;
for ((i=3;i<=$#;i++))
do
        add=${args[$i]};
        irrelevant_modules=`echo $irrelevant_modules $add`;

done

local b;

#b=$(Remove_parts_string "$a" "cms.InputTag" "(" ")" "," "="  $irrelevant_modules | tr -d " \" " |tr ' ' '\n')
#echo "$b"


#b=$(Remove_parts_string "$a"  "\"" "'"   $irrelevant_modules  | tr ',' ' ' |tr ' ' '\n'  );
local aaaa;
aaaa=`echo "$a" | tr -d "\"" | tr -d "'"`;
b=$( Remove_parts_string "$aaaa" $irrelevant_modules  | tr ',' ' ' |tr ' ' '\n'  );
echo "$b" | uniq;



#HLT_MODULE_end_
}

Remove_parts_string()
{

#Remove_parts_string_begin_	
local help='
#$1 -- string
#$2 -- word1 
#$3 -- word2
etc to remove 
'

if [ $# -lt 1 ]; then print_help "$0" "$help"; return 1; fi

local args;
args=("$@");
local remove;
local string;

string="$1";

#echo "string = $string";
local aaa;
aaa=($( echo $string));
local bbb;
#echo "aaa=${#aaa[@]}";

local outstring
outstring="";


for ((j=1; j<=${#aaa[@]}; j++))
do

for ((i=2;i<=$#;i++))
do
	remove=${args[$i]};
#	echo remove=$remove
#	echo index=$j
#	bbb=`expr index "$j" "$remove"`;	
	bbb=`echo "${aaa[$j]}" | grep  "$remove"`;	 
#	echo bbb=$bbb
#	echo remove=$remove
	
	if [ -n "$bbb"  ];  then aaa[$j]="";fi;

#	echo j=${aaa[$j]}

	#string=`echo $string | sed -ne "s/$remove//pg"`
#	string=`echo ${string//"$remove"/}`

#	echo  "${args[$i]}" 
#	echo strin=$string
	done 
done

for ((j=1; j<=${#aaa[@]}; j++))
do
	outstring=`echo $outstring ${aaa[$j]}`;

done
#echo $string
echo $outstring

#Remove_parts_string_end_
}


Extract_some_part()
{
#Extract_some_part_begin_	
local help='
#$1 -- file to be processed
#$2 -- word1 in the begin line
#$3 -- word2 in the begin line
	position of word1 must be < postion of word2
#$4 -- word in the  end line
#$5 -- position of the word in the end line

That is the part of processing HLT_MENU.cff file

'

if [ $# -lt 5 ]; then print_help "$0" "$help"; return 1; fi

#beg=`findLineNumber "$2" "$3" "$1"`
#end=`findLineNumber "$4" "$5" "$1"`
#echo $beg
#echo $end
#sed -ne "$beg,${end}p" $1 >  $$_file_tmp;

#awk 'BEGIN{ found=0 };  { if (index($0,"hltBLifetimeL3BJetTagsbbPhiL1FastJetFastPV")==1 && match($0,"=")>0) {found=1}; if(found>0) print $0; if (index($0,")")==1)  found=0;} END{ print found}' HLT_GRun_cff.py



awk "BEGIN{ found=0 };  { if (index(\$0,\"$2\")>0 &&  index(\$0,\"$3\")>0 && index(\$0,\"$2\")< index(\$0,\"$3\") && index(\$0,\"#\") == 0 ) {found=1}; if(found>0) print \$0; if (index(\$0,\"$4\")==$5)  found=0;}" $1



#cat    $$_file_tmp;
#rm $$_file_tmp;

#Extract_some_part_end_
}

ExtractFunctionBody()
{
#ExtractFunctionBody_begin_
local help='
#$1 -- file to be processed
#$2 -- name of function
'

if [ $# -lt 2 ]; then print_help "$0" "$help"; return 1; fi

local beg=`findLineNumber "$2" "_begin_" "$1" `
local end=`findLineNumber "$2" "_end_" "$1"` 
sed -ne "$beg,${end}p" $1 >  $$_file_tmp;
echo "$2()";
echo "{";
cat    $$_file_tmp;
echo "}";

rm $$_file_tmp;


#ExtractFunctionBody_end_
}


Comment()
{
#Comment_begin_

local help='
#$1 -- file to be modified
#$2 -- from to be commented
#$3 -- until to be commented
#$4 --symbol of comments (optional)
'

if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi

if [ -n $2 -a -n $1 -a -n $3 ]
	then
		local tmp_file="$$_file_tmp";
		local tmp_file2="$$_file_tmp2";
		cp $1 $tmp_file;

		if [ -z $4 ]
		 then
		 awk "BEGIN{act=0} { if (match(\$0,\"$3\")) act=0; if (act==0) print \$0; if (act>0) { print \"#\"\$0 }; if (match(\$0,\"$2\")) act=1;    }" $tmp_file > $tmp_file2;
		else
		 awk "BEGIN{act=0} { if (match(\$0,\"$3\")) act=0; if (act==0) print \$0; if (act>0) { print \"$4\"\$0 }; if (match(\$0,\"$2\")) act=1;    }" $tmp_file > $tmp_file2;
		fi		
		mv $tmp_file2 $1;
		rm $tmp_file;
fi
#Comment_end_ 
}

UnComment()
{
# UnComment_begin_
local help='
#$1 -- file to be modified
#$2 -- from to be uncommented
#$3 -- until to be uncommented
#$4 --symbol of comments (optional)
'


if [ $# -lt 3 ]; then print_help "$0" "$help"; return 1; fi

if [ -n $2 -a -n $1 -a -n $3 ]
	then
	local tmp_file="$$_file_tmp";
	 local tmp_file2="$$_file_tmp2";
	cp $1 $tmp_file;
	 if [ -z $4 ]
                 then
	awk "BEGIN{act=0} {if (match(\$0,\"$3\")) act=0;if (act==0) print \$0;  if (act>0) { str=\$0; sub(\"#\",\"\",str); print str}; if (match(\$0,\"$2\")) act=1;}" $tmp_file > $tmp_file2;
	else
	awk "BEGIN{act=0} {if (match(\$0,\"$3\")) act=0;if (act==0) print \$0;  if (act>0) { str=\$0; sub(\"$4\",\"\",str); print str}; if (match(\$0,\"$2\")) act=1;}" $tmp_file > $tmp_file2;

	fi	
		 mv $tmp_file2 $1;
                rm $tmp_file;
fi
# UnComment_end_
}


UnCommentConcrete()
{
#UnCommentConcrete_begin_ 
local help='
#$1 -- file to be modified
#$2 -- from to be uncommented
#$3 -- until to be uncommented
#$4 -- what to uncomment
#$5 -- what to be not presented at the string
'


if [ $# -lt 5 ]; then print_help "$0" "$help"; return 1; fi


	local tmp_file="$$_file_tmp";
 	local tmp_file2="$$_file_tmp2";
        cp $1 $tmp_file;

if [ -z "$5" ]
	then
	awk "BEGIN{act=0} {if (match(\$0,\"$3\")) act=0;if (act==0) print \$0;  if (act>0 && match(\$0,\"$4\")) { str=\$0; sub(\"#\",\"\",str); print str}; if (match(\$0,\"$2\")) act=1;}" $tmp_file > $tmp_file2;
fi

if [ -n "$5" ] 
	then
	awk "BEGIN{act=0} {if (match(\$0,\"$3\")) act=0;if (act==0) print \$0;  
	if (act>0){ if (match(\$0,\"$4\") && !match(\$0,\"$5\")){
	  str=\$0; sub(\"#\",\"\",str); print str}; if (!(match(\$0,\"$4\") && !match(\$0,\"$5\")))  print  \$0 }; if (match(\$0,\"$2\")) act=1;}" $tmp_file >  $tmp_file2;
fi

		 mv $tmp_file2 $1;
                rm $tmp_file;

#UnCommentConcrete_end_ 
}



