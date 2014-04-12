<? include "configure.php"?>
<html>
<head>
  <meta name="AUTHOR" content="M. De Mattia" />
  <meta http-equiv="content-type" content="text/html; charset=utf-8" />
  <title>HDQM Web Interface Home Page</title>
  <style type="text/css" title="currentStyle">
    @import "/<?=$GLOBALS['userName']?>/lib/dataTables/media/css/demo_page.css";
    @import "/<?=$GLOBALS['userName']?>/lib/dataTables/media/css/demo_table_jui.css";
  </style>
  <style>
    #dt_example{
      padding: 20px;
    }
  </style>
<link rel="stylesheet" href="http://ajax.googleapis.com/ajax/libs/jqueryui/1.7.2/themes/smoothness/jquery-ui.css" type="text/css" />
</head>
<body id="dt_example">
  <h1 id="title">HDQM Web Interface</h1>
<div align="right"><a href="http://indico.cern.ch/materialDisplay.py?contribId=3&sessionId=3&materialId=slides&confId=82611" target="_blank">Presentation Slides</a></div>
<!--<div align="right"><a href="#" target="_new">User Manual</a></div>-->
  <ul>
    <li> <a href="./WebInterface.php?subDet=Tracking&tagVersion=V2" name="Tracking Interface">Tracking Interface</a> </li>
    <li> <a href="./WebInterface.php?subDet=SiStrip&tagVersion=V2" name="SiStrip Interface">SiStrip Interface</a> </li>
    <li> <a href="./WebInterface.php?subDet=SiPixel&tagVersion=V2" name="SiPixel Interface">SiPixel Interface</a> </li>
    <li> <a href="./WebInterface.php?subDet=RPC&tagVersion=V1" name="RPC Interface">RPC Interface</a> </li>
  </ul>
</body>
</html>
