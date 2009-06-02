<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php 
echo "<title>bad channels selection page - ecalod-web01</title>\n";
?>
</script>

</head>

<body>

<h1><center>
<?php
echo "<font color=blue>bad channels selection criteria - ecalod-web01</font>\n";
?>
</center></h1>
<br/>

<hr>

</p>
<form method=get action="eb_bad_channels_lists.php" name="run">

<h2>Barrel</h2>

Site:
<select size="1" name="site">
<option value="P5_Co">P5_Co</option>
<option value="P5">P5</option>
</select>

Run number:
<input type=text size=10 name="run" value="99999">

Channel Status:
<select size="1" name="status">
<option value="1">bad</option>
<option value="0">good</option>
</select>

<p>

<input type=submit size=5 value="Go">
<input type=reset size=5 value="Reset">
</form><br>

<hr>

</p>
<form method=get action="ee_bad_channels_lists.php" name="run">

<h2>Endcap</h2>

Site:
<select size="1" name="site">
<option value="P5_Co">P5_Co</option>
<option value="P5">P5</option>
</select>

Run number:
<input type=text size=10 name="run" value="99999">

Channel Status:
<select size="1" name="status">
<option value="1">bad</option>
<option value="0">good</option>
</select>

<p>

<input type=submit size=5 value="Go">
<input type=reset size=5 value="Reset">
</form><br>

<hr>

<p>
<font size=1>Benigno, Fabio  &amp; Giuseppe, CMS-Trieste, 2006/10/18</font><br> 

</body>
</html>


