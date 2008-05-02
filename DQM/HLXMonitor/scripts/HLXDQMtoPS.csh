#!/bin/tcsh

set file = `ls $1`
set OutputDir = "/cms/mon/data/dqm/lumi"
set OutPrefix = "HLXDQM"
set psDir = "$OutputDir/ps"

set Run = $2
set LS = $3
set suffix = _${Run}_${LS}

mkdir -p $OutputDir
mkdir -p $psDir

rm -f /tmp/TempScript.C

if ( `echo $file` == '' ) then
    echo "$1 does not exist"
else

cat >> /tmp/TempScript.C <<EOF
{    
    gROOT->SetStyle("Plain");

    TFile myfile("$1","Read");

    TCanvas HLXC("HLXC");
    HLXC->Divide(3,3);

    if(myfile.cd("DQMData")){
	if(myfile.cd("DQMData/HFPlus")){
EOF

foreach n (`seq 1 18`)

set prefix = 'HF_Plus_Wedge"$n"_'

cat >> /tmp/TempScript.C <<EOF

    if(myfile.cd("DQMData/HFPlus/Wedge$n")){
	HLXC.cd(1);
	`echo $prefix`ETSum->Draw();
	HLXC.cd(4);
	`echo $prefix`Set1_Above->Draw();
	HLXC.cd(5);
	`echo $prefix`Set1_Between->Draw();
	HLXC.cd(6);
	`echo $prefix`Set1_Below->Draw();
	HLXC.cd(7);
	`echo $prefix`Set1_Above->Draw();
	HLXC.cd(8);
	`echo $prefix`Set1_Between->Draw();
	HLXC.cd(9);
	`echo $prefix`Set1_Below->Draw();
	HLXC.Print("$psDir/HFPlusWedge$n.ps");
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
	HLXC.cd(1);
	`echo $prefix`ETSum->Draw();
	HLXC.cd(4);
	`echo $prefix`Set1_Above->Draw();
	HLXC.cd(5);
	`echo $prefix`Set1_Between->Draw();
	HLXC.cd(6);
	`echo $prefix`Set1_Below->Draw();
	HLXC.cd(7);
	`echo $prefix`Set1_Above->Draw();
	HLXC.cd(8);
	`echo $prefix`Set1_Between->Draw();
	HLXC.cd(9);
	`echo $prefix`Set1_Below->Draw();
	HLXC.Print("$psDir/HFPlusWedge$n.ps");
    }

EOF

end

cat >> /tmp/TempScript.C <<EOF
	}
    } 
}
EOF

root -b -q .x /tmp/TempScript.C

#rm /tmp/TempScript.C

gs -sDEVICE=pswrite -sOutputFile=$OutputDir/$OutPrefix$suffix.ps -dNOPAUSE -dBATCH $psDir/*.ps

endif

