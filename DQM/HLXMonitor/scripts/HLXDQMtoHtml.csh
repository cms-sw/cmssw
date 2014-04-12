#!/bin/tcsh

set file = `ls $1`
set picExt = "gif"
set Run = $2
set LS = $3
set OutputDir = "/cms/mon/data/dqm/lumi/dqmhtml/$Run/$LS"
set OutputHTMLDirectory = "http://cmsmon.cern.ch/lumi/dqmhtml/$Run/$LS"
set OutputHTMLINK = "/cms/mon/data/dqm/lumi/dqmhtml"
set OutPrefix = "HLXDQM"


set suffix = .`echo $Run`.`echo $LS`

mkdir -p $OutputDir
mkdir -p $OutputDir/$picExt

#set filename = $OutPrefix$suffix.html

set filename = $OutputHTMLINK/index.html


if !(-e $filename) then 

cat > $filename  <<EOF
<html>
<head>                                                                         
<title>
</title>
</head>
<body>
<h2> This File Contains the List of DQM file plots for Run $Run and all Lumi Sections </h2>                                                     
<p>

<ul>
EOF

echo "$filename exist now"

endif

rm -f /tmp/TempScript.C

if ( `echo $file` == '' ) then
    echo "$1 does not exist"
else

cat >> /tmp/TempScript.C <<EOF
{
    
    gROOT->SetStyle("Plain");

    TFile myfile("$1","Read");
    
    if(myfile.cd("DQMData")){
	if(myfile.cd("DQMData/HFPlus")){
EOF

foreach n (`seq 1 18`)

set prefix = 'HF_Plus_Wedge"$n"_'

cat >> /tmp/TempScript.C <<EOF

    if(myfile.cd("DQMData/HFPlus/Wedge$n")){
	`echo $prefix`ETSum->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`ETSum$suffix.$picExt");
	`echo $prefix`Set1_Above->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set1_Above$suffix.$picExt");
	`echo $prefix`Set1_Between->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set1_Between$suffix.$picExt");
	`echo $prefix`Set1_Below->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set1_Below$suffix.$picExt");
	`echo $prefix`Set1_Above->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set2_Above$suffix.$picExt");
	`echo $prefix`Set1_Between->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set2_Between$suffix.$picExt");
	`echo $prefix`Set1_Below->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set2_Below$suffix.$picExt");
    }
EOF

end

cat >> /tmp/TempScript.C <<EOF
 }
 if(myfile.cd("DQMData/HFMinus")){

EOF

foreach n (`seq 19 36`)

set prefix = 'HF_Minus_Wedge"$n"_'

cat >> /tmp/TempScript.C <<EOF

    if(myfile.cd("DQMData/HFMinus/Wedge$n")){
	`echo $prefix`ETSum->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`ETSum$suffix.$picExt");
	`echo $prefix`Set1_Above->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set1_Above$suffix.$picExt");
	`echo $prefix`Set1_Between->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set1_Between$suffix.$picExt");
	`echo $prefix`Set1_Below->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set1_Below$suffix.$picExt");
	`echo $prefix`Set1_Above->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set2_Above$suffix.$picExt");
	`echo $prefix`Set1_Between->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set2_Between$suffix.$picExt");
	`echo $prefix`Set1_Below->Draw();
	c1->SaveAs("$OutputDir/$picExt/`echo $prefix`Set2_Below$suffix.$picExt");
    }
EOF

end

cat >> /tmp/TempScript.C <<EOF
	}
    } 
}
EOF

root -b -q .x /tmp/TempScript.C

rm /tmp/TempScript.C


# Create html

cat >> $OutputDir/$OutPrefix$suffix.html <<EOF
<html>
<head>
<title> </title>
</head>
<body>
<h1>Run $Run - Lumi Section $LS</h1>
<h2>HF Plus</h2>
EOF

foreach n (`seq 1 18`)

set prefix = 'HF_Plus_Wedge"$n"_'

cat >> $OutputDir/$OutPrefix$suffix.html <<EOF
    <hr>
    <h3>HF Plus - Wedge $n</h3>
    <a href="$OutputHTMLDirectory/$picExt/`echo $prefix`ETSum$suffix.$picExt"><img src="$OutputHTMLDirectory/$picExt/`echo $prefix`ETSum$suffix.$picExt"></a></br>
  <a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Above$suffix.$picExt">  <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Above$suffix.$picExt" width="30%"></a>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Between$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Between$suffix.$picExt" width="30%"></a>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Below$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Below$suffix.$picExt" width="30%"></a></br>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Above$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Above$suffix.$picExt" width="30%"></a>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Between$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Between$suffix.$picExt" width="30%"></a>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Below$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Below$suffix.$picExt" width="30%"></a></br>
EOF

end

cat >> $OutputDir/$OutPrefix$suffix.html <<EOF
    <h2>HF Minus</h2>  
EOF

foreach n (`seq 19 36`)

set prefix = 'HF_Minus_Wedge"$n"_'

cat >> $OutputDir/$OutPrefix$suffix.html <<EOF
    <hr>
    <h3>Wedge $n</h3>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`ETSum$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`ETSum$suffix.$picExt"></br>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Above$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Above$suffix.$picExt" width="30%">
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Between$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Between$suffix.$picExt" width="30%">
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Below$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set1_Below$suffix.$picExt" width="30%"></br>
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Above$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Above$suffix.$picExt" width="30%">
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Between$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Between$suffix.$picExt" width="30%">
<a href="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Below$suffix.$picExt">    <img src="$OutputHTMLDirectory/$picExt/`echo $prefix`Set2_Below$suffix.$picExt" width="30%"></br>
EOF

end

cat >> $OutputDir/$OutPrefix$suffix.html <<EOF
</body>
</html>
EOF


cat >> $OutputHTMLINK/index.html <<EOF
<li> <a href="$OutputHTMLDirectory/$OutPrefix$suffix.html"> Run $Run LS $LS</a>

</body>
</html>
EOF


endif

