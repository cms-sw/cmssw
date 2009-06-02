<?php

if (array_key_exists("wherereq", $_GET)) $wherereq = $_GET["wherereq"];
else $wherereq = "";

?>
<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php
echo "<title>bad channels list - ecalod-web01</title>\n";
?>
<link REL="STYLESHEET" TYPE="text/css" HREF="resources.css">

<script language="JavaScript">
<!-- hide from old browsers

function makeArray(n) {
  this.length = n;
  for ( i=0; i<n; i++ ) {
    this[i]=0;
  }
  return this;
}

function displayDate() {
  var this_month = new makeArray(12);
  this_month[0]  = "January";
  this_month[1]  = "February";
  this_month[2]  = "March";
  this_month[3]  = "April";
  this_month[4]  = "May";
  this_month[5]  = "June";
  this_month[6]  = "July";
  this_month[7]  = "August";
  this_month[8]  = "September";
  this_month[9]  = "October";
  this_month[10] = "November";
  this_month[11] = "December";
  var time = new Date();
  var year   = time.getYear();
  if( year < 1900 ) {
    year += 1900;
  }
  var month  = time.getMonth();
  var day    = time.getDate();
  var hour   = time.getHours();
  var minute = time.getMinutes();
  var second = time.getSeconds();
  var temp = "" + ((hour<10)?"0":"") + hour;
  temp += ((minute<10)?":0":":") + minute;
  temp += ((second<10)?":0":":") + second;
  temp += ",    "+day+" "+this_month[month]+" "+year;
  return( temp );
}

//-->
</script>

</head>
<body>

<?php
  require(".ht_php_utils.php");
?>

<?php

$site = $_REQUEST['site'];
$run = $_REQUEST['run'];
$status = $_REQUEST['status'];

echo "<h1><center><font color=blue>List of bad channels for run ".$run." (ecalod-web01)</font></center></h1><br><hr>\n";
echo "<hr>\n";

?>

<?php

# =======================================
# now we use a program to set the list...

$string = array();
$command = "env TNS_ADMIN=/etc /var/www/html/badchannels/scripts/ee-list_bad_channels.pl ".$site." ".$run." ".$status;
exec( $command, $string );

#========================================

?>

<pre>
<?php

for( $k=0; $k<(sizeof($string)-1); $k++ ) {
  tr_alt();
  echo $string[$k];
  echo "\n";
}

?>
</pre>

<hr>
<hr>
<font face="Arial, Helvetica" color=black size=1>
Benigno &amp; Giuseppe, CMS-Trieste, 2007/04/21<br>
</font>

</body>
</html>



