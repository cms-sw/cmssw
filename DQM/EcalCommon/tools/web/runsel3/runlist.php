<?php

if (array_key_exists("wherereq", $_GET)) $wherereq = $_GET["wherereq"];
else $wherereq = "";

$datadir = "/var/www/html/logs";

?>
<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php
echo "<title>runs list - ecalod-web01</title>\n";
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

<meta http-equiv="Refresh" content="300">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Cache-Control" content="no-cache">

</head>
<body>

<?php
  require(".ht_php_utils.php");
?>

<?php

$parsercommand = "./parser/parser ".$datadir;
$selection = "";

$runtype = $_REQUEST['runtype'];
if ( $runtype != "any" ) {
  $selection .= " and Run Type = ".$runtype;
  $parsercommand .= " -t ".$runtype;
}

$rungt = $_REQUEST['rungt'];
if ( $rungt != "-" ) {
  $selection .= " and Run >= ".$rungt;
  $parsercommand .= " -R ".$rungt;
}

$runlt = $_REQUEST['runlt'];
if ( $runlt != "-" ) {
  $selection .= " and Run <= ".$runlt;
  $parsercommand .= " -r ".$runlt;
}

# we remove the first "and"
$selection = substr($selection, 4);

$pagesize = $_REQUEST['pagesize'];
$sortorder = $_REQUEST['sortorder'];
$first = $_REQUEST['first'];
if( "$first" == "" ) $first = 0;

$parsercommand .= " -f ".$first;
$parsercommand .= " -p ".$pagesize;

if     ( $sortorder == "rdw" )  $parsercommand .= " -o rundown";
else if( $sortorder == "rup" )  $parsercommand .= " -o runup";

# store the options
$allopt .= "runtype=".$runtype;
$allopt .= "&rungt=".$rungt;
$allopt .= "&runlt=".$runlt;
$allopt .= "&sortorder=".$sortorder;

if( "$selection" == "" ) {
  echo "<h1><center><font color=blue>List of all runs (ecalod-web01)</font></center></h1><br>\n";
}
else {
  echo "<h1><center><font color=blue>List of selected runs (ecalod-web01)</font></center></h1><br><hr>\n";
  $selection = stripslashes($selection);
  echo "<h3>selection criteria: <code>$selection</code><br></h3>\n";
  echo "<hr>\n";
}

?>

<?php

# =======================================
# now we use a program to set the list...

$string = array();
exec( $parsercommand, $string );

#========================================

$entries = $string[ sizeof( $string ) - 1 ]; 

if( $pagesize == 0 ) $pagesize = $entries;
if( $first >= $entries )  $first = $first-$pagesize; 
$last = $first + $pagesize;
if( $last > $entries ) $last = $entries;
$next = $first + $pagesize;
$prev = $first - $pagesize; 
if( $prev < 0 ) $prev = 0;
$end  = $entries - $pagesize;

if( $pagesize < $entries ) {
  echo "<hr>\n";
  echo "<font face=\"Arial, Helvetica\"><b>Entries: ".$entries."</b>&nbsp;<font color=blue size=2>(Now displaying from ".($first+1)." to $last)</font></font><br>\n";
  echo "<hr>\n"; 
  if( $first != 0 ) {
    echo "<a href=\"runlist.php?".$allopt."&first=0&pagesize=".$pagesize."\"><img src=sb.gif border=0></a>&nbsp;&nbsp;&nbsp;&nbsp;";
    echo "<a href=\"runlist.php?".$allopt."&first=".$prev."&pagesize=".$pagesize."\"><img src=sp.gif border=0></a>&nbsp;&nbsp;&nbsp;&nbsp;";
  }
  if( $last != $entries ) {
    echo "<a href=\"runlist.php?".$allopt."&first=".$next."&pagesize=".$pagesize."\"><img src=sn.gif border=0></a>&nbsp;&nbsp;&nbsp;&nbsp;";
    echo "<a href=\"runlist.php?".$allopt."&first=".$end."&pagesize=".$pagesize."\"><img src=se.gif border=0></a>&nbsp;&nbsp;&nbsp;&nbsp;";
  }
  echo "<hr>\n"; 
}
else {
  echo "<hr>\n";
  echo "<b>Entries: ".$entries."</b><br>\n";
}

?>

<table class="listentries" width=1000 border=1 bordercolor=black rules=all cellpadding=2 cellspacing=0>

<tr>
<th class="entry">Entry</th>
<th class="type">Type</th>
<th class="date"><nobr>Proc. Date</nobr></th>
<th class="time"><nobr>Proc. Time</nobr></th>
<th class="runnb">Number</th>
<th class="log">Log</th>
<th class="gui">Gui</th>
</tr>

<?php

function findinside($start, $end, $string) {
  preg_match_all('/' . preg_quote($start, '/') . '([^\.)]+)'. preg_quote($end, '/').'/i', $string, $m);
  return $m[1];
}

for( $k=0; $k<(sizeof($string)-1); $k++ ) {
  tr_alt();
  echo "\n";

#  echo $string[$k];
#  echo "\n";

  $run = findinside("<td class=runnb>", "</td>", $string[$k]);

  $type=array();
  exec("env TNS_ADMIN=/etc /var/www/html/runsel3/runtype.pl $run[0]", $type);

  $line = str_replace("<td class=type> <font color=red><nobr>&lt;not available&gt;</nobr></font> </td>", "<td class=type><nobr>$type[0]</td>", $string[$k]);

  echo $line;
  echo "\n";
}

?>

</table>

<hr>
<font face="Arial, Helvetica" color=black size=1>
This page automatically updates every 5 minutes. <font color=blue>Last update:
<script language="JavaScript">document.write(displayDate());</script><br></font>
Benigno, Fabio &amp; Giuseppe, CMS-Trieste, 2007/06/21<br>
</font>

</body>
</html>



