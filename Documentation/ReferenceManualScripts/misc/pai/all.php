
<?php

$aawgs = array("B2G", "BPH", "BTV", "CFT", "DIF", "EGM", "EWK", "EXO", "FSQ", "FTR", "FWD", "GEN", "HIG", "HIN", "JME", "LUM", "MUO", "PFT", "PRF", "QCD", "SBM", "SMP", "SUS", "TAU", "TOP", "TRG", "TRK");
$years = array("2007","2008","2009","2010","2011","2012","2013");

$array = "array(";

foreach($aawgs as $awg){
	foreach($years as $year){	
		echo "<a href='http://cms.cern.ch/iCMS/analysisadmin/analysismanagement?awg=".$awg."&dataset=All&awgyear=".$year."' style='font-family:monospace,Courier'>".$awg." ".$year."</a>, ";
		$array .= "'http://cms.cern.ch/iCMS/analysisadmin/analysismanagement?awg=".$awg."&dataset=All&awgyear=".$year."', ";
	}
	echo "<br/>";
}

$array.=");";

echo $array;

?>