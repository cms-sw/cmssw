<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php
echo "<title>runs status on ecalod-web01</title>\n";
?>
<!-- <link REL="STYLESHEET" TYPE="text/css"> -->

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

<meta http-equiv="Refresh" content="300">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Cache-Control" content="no-cache">

</head>
<body>

<?php
  require(".ht_php_utils.php");
?>

<?php
echo "<p><center><font face=\"Arial, Helvetica\" color=blue size=6><b>RUN status on ecalod-web01</b><br></font></center>";
?>

<hr>
<font face="Arial, Helvetica" color=black size=2>This page automatically updates every 5 minutes. If that time looks too long for you... Just click "reload" on your browser...</font><br> 
<font face="Arial, Elvetica size=4" color=blue>Last update:<b>
<script language="JavaScript">document.write(displayDate());</script></b><br></font>
<hr>

<table class="listentries" width=700 border=1 bordercolor=black rules=all cellpadding=2 cellspacing=0>

<tr>
<th bgcolor=turquoise><font face="Arial, Helvetica">Run</font></th>
<th bgcolor=turquoise><font face="Arial, Helvetica">Type</font></th>
<th bgcolor=turquoise><font face="Arial, Helvetica">Status</font></th>
<th bgcolor=turquoise><font face="Arial, Helvetica">Date</font></th>
<th bgcolor=turquoise><font face="Arial, Helvetica">Time</font></th>
<th bgcolor=turquoise><font face="Arial, Helvetica">File</font></th>
<!-- 
<th bgcolor=turquoise><font face="Arial, Helvetica">CMSSW</font></th>
<th bgcolor=turquoise><font face="Arial, Helvetica">Problems</font></th>
<th bgcolor=turquoise><font face="Arial, Helvetica">Log</font></th>
 -->
</tr>

<?php

$string = array();
$command = "./runstat.sh";
exec( $command, $string );

for( $i=0; $k<sizeof($string) && strpos( $string[$i], "Summary" ) === FALSE; $i++ ) {
  if( strpos( $string[$i], "-----") === FALSE && strpos( $string[$i], "Run") === FALSE ) {
    echo "<tr>";
    $pieces = explode( " ", $string[$i] );
    $counter = 0;
    for( $j=0; $j<sizeof( $pieces); $j++ ) {
      $len = strlen( $pieces[$j] );
      if( $len > 0 ) {
	$counter = $counter + 1;
	if( $counter == 3 ) {
	  if( "$pieces[$j]" == "daq_fail" || "$pieces[$j]" == "daq_empty" ) {
	    echo "<td align=center><font face=\"Arial, Helvetica\" color=\"red\">$pieces[$j]</font></td>";
	  }
	  elseif ( "$pieces[$j]" == "daq_done" ) {
	    echo "<td align=center><font face=\"Arial, Helvetica\" color=\"orange\">$pieces[$j]</font></td>";
	  }
	  elseif ( "$pieces[$j]" == "mon_done" ) {
	    echo "<td align=center><font face=\"Arial, Helvetica\" color=\"green\">$pieces[$j]</font></td>";
	  }
	  elseif ( "$pieces[$j]" == "daq_run" || "$pieces[$j]" == "mon_run" ) {
	    echo "<td align=center><font face=\"Arial, Helvetica\" color=\"blue\">$pieces[$j]</font></td>";
	  }
	  elseif ( "$pieces[$j]" == "mon_fail" ) {
	    echo "<td align=center><font face=\"Arial, Helvetica\" color=\"darkred\">$pieces[$j]</font></td>";
	  }
	  else {
	    echo "<td align=center><font face=\"Arial, Helvetica\">$pieces[$j]</font></td>";
	  }
	}
	elseif( $counter == 4 ) {
	  $d = $pieces[$j];
	  $d = substr( $d, 0, 4 )."/".substr( $d, 4, 2 )."/".substr( $d, 6, 2 );
	  echo "<td align=center><font face=\"Arial, Helvetica\">$d</font></td>";
	}
	elseif( $counter == 5 ) {
	  $t = $pieces[$j];
	  $t = substr( $t, 0, 2 ).":".substr( $t, 2, 2 ).":".substr( $t, 4, 2 );
	  echo "<td align=center><font face=\"Arial, Helvetica\">$t</font></td>";
	}
	else {
	  echo "<td align=center><font face=\"Arial, Helvetica\">$pieces[$j]</font></td>";
	}
      }
    }
  }
}

?>

</table>

<hr>
<p>
<font face="Arial, Helvetica" color=black size=1>
Another stuff by CMS-Trieste, 2007/04/10</font><br> 
</font>

</body>
</html>



