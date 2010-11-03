<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php
$detector = $_REQUEST['detector'];
echo "<h1><center><font color=blue>ECAL DQM Reset Page - ecalod-web01</font></center></h1><br><hr>\n";
?>

</head>
<body>

<?php
$file = "/data/ecalod-disk01/dqm-data/reset/".$detector."_test";

while ( file_exists("$file") ==  "TRUE" ) {
  sleep(2);
}

echo "<hr>\n";
echo "<center>\n";
echo "<h2>Reset of ECAL DQM histograms for ".$detector." done !</h2>\n";
echo "</center>\n";
echo "<hr>\n";

?>

<hr>
<font face="Arial, Helvetica" color=black size=1>
<font size=1>Eamnuele &amp; Giuseppe, CMS-Trieste, 2010/11/03</font><br>
</font>

</body>
</html>
