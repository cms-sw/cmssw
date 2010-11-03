<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php
$detector = $_REQUEST['detector'];
echo "<h1><center><font color=blue>ECAL DQM Reset Page - ecalod-web01</font></center></h1><br><hr>\n";
?>

<script language="JavaScript1.1">
<!--
var sURL = unescape(window.location.pathname);
function doLoad() { setTimeout( "refresh()", 10 ); }
<?php
echo 'function refresh() { window.location.replace("dqm_reset_wait.php?detector='.$detector.'"); }'
?>
//-->
</script>

</head>
<body onload="doLoad()">

<?php
$file = "/data/ecalod-disk01/dqm-data/reset/".$detector."_test";

if ( file_exists("$file") ==  "TRUE" ) {
  echo "<hr>\n";
  echo "<center>\n";
  echo "<h2>I already said, you have to wait ... </h2>\n";
  echo "<h2>(up to 5-10 minutes)</h2>\n";
  echo "</center>\n";
  echo "<hr>\n";
} else {
  exec("touch ".$file);
  exec("chmod a+rw ".$file);
  echo "<hr>\n";
  echo "<center>\n";
  echo "<h2>Reset of ECAL DQM histograms for ".$detector." is in progress ... please wait.</h2>\n";
  echo "</center>\n";
  echo "<hr>\n";
}

?>

<hr>
<font face="Arial, Helvetica" color=black size=1>
<font size=1>Eamnuele &amp; Giuseppe, CMS-Trieste, 2010/11/03</font><br>
</font>

</body>
</html>
