<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php 
echo "<title>run selection page - ecalod-web01</title>\n";
?>
<link REL="STYLESHEET" TYPE="text/css" HREF="resources.css">
</script>

</head>

<body>

<h1><center>
<?php
echo "<font color=blue>Run selection criteria - ecalod-web01</font>\n";
?>
</center></h1>
<br/>

<hr>

<?php
  require(".ht_php_utils.php");
?>

<p><b><font color=blue>
An <font color=red>AND</font> of all the activated criteria is used to select the runs.
</font></b><br>

</p>
<form method=get action="runlist.php" name="select_runs">

<ul type=disk>

<li><font face="Arial, Helvetica">Run Type:</font>
<select size="1" name="runtype">
<option value="any">any type</option>
<option value="LASER">laser</option>
<option value="TEST_PULSE">testpulse</option>
<option value="PEDESTAL">pedestal</option>
<option value="COSMIC">cosmic</option>
<option value="NOTAVAILABLE">not available</option>
<option value="UNKNOWN">unknown</option>
</select>

<li><font face="Arial, Helvetica">Run number greater or equal to:</font>
<input type=text size=10 name="rungt" value="-">

<li><font face="Arial, Helvetica">Run number less or equal to:</font>
<input type=text size=10 name="runlt" value="-"><br>

</ul>

<font face="Arial, Helvetica" color=blue><b>
Entries per page:
<select size="1" name="pagesize">
<option value="25">25</option>
<option value="50">50</option>
<option value="100">100</option>
<option value="500">500</option>
<option value="0">all</option>
</select>
</b></font> 

&nbsp;

<font face="Arial, Helvetica" color=blue><b>
Sorting order:
<select size="1" name="sortorder">
<option value="rdw">by decreasing run number</option>
<option value="rup">by increasing run number</option>

</select>
</b></font> 


<p>

<input type=submit size=5 value="Go">
<input type=reset size=5 value="Reset">
</form><br>

<hr>

<p>
<font size=1>Benigno, Fabio  &amp; Giuseppe, CMS-Trieste, 2007/06/21</font><br> 

</body>
</html>


