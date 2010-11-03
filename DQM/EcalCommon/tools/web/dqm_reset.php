<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>

<?php 
echo "<title>ECAL DQM Reset Page - ecalod-web01</title>\n";
?>

</head>
<body>

<h1><center>
<?php
echo "<font color=blue>ECAL DQM Reset Page - EXPERTS ONLY - ecalod-web01</font>\n";
?>
</center></h1>
<br/>

<hr>

</p>
<form method=get action="dqm_reset_touch.php" name="run">

<p>
<center>
<form>
<input style="border-style:outset; font-size:30px; font-weight:bold; color:#FF0000;" type="button" value="Push to Reset ECAL Barrel DQM histograms" onclick="window.location.href='dqm_reset_touch.php?detector=EB'">
</form>
</center>
<p>

<hr>

<p>
<center>
<form>
<input  style="border-style:outset; font-size:30px; font-weight:bold; color:#FF0000;" type="button" value="Push to Reset ECAL Endcap DQM histograms" onclick="window.location.href='dqm_reset_touch.php?detector=EE'">
</form>
</center>
<p>

<hr>

<p>
<font size=1>Eamnuele &amp; Giuseppe, CMS-Trieste, 2010/11/03</font><br> 

</body>
</html>
