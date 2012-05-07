<?php

        //$doxy_URL = "http://cms-service-sdtweb.web.cern.ch/cms-service-sdtweb/doxygen/";
        //$doxy_PATH = "/afs/cern.ch/cms/sdt/web/doxygen/";
        $doxy_URL = "http://cmssdt.cern.ch/SDT/doxygen/";
        $doxy_PATH = "/data/doxygen/";
	
	
	$output = "";
	$object = "";
	$object_name = "";
	$grep = "";
	$star = "";
	$ext = "\.html";

	$r = isset($_GET["r"])?$_GET["r"]:""; // release
	$c = isset($_GET["c"])?$_GET["c"]:""; // class
	$s = isset($_GET["s"])?$_GET["s"]:""; // struct
	$n = isset($_GET["n"])?$_GET["n"]:""; // namespace
	$o = isset($_GET["o"])?$_GET["o"]:""; // source
	$k =  isset($_GET["k"])?$_GET["k"]:""; // keywords
        $K =  isset($_GET["K"])?$_GET["K"]:""; // keywords strict // changed: 10-07-08
	
	
	if ($K != "" && $k != "")	// changed: 10-07-08
	{
		print "K and k can't be specified at the same time !!!";
		exit;
	}
		
	if ($K != "") $k = $K;	// changed: 10-07-08

	if ($k != "")
	{
		$k = trim($k,"{");
		$k = trim($k,"}");
		$k = trim($k);
		if ($k != "")
		{
			$k = str_replace("*",".*",$k);
			$keywords = explode(",", $k);   // changed: 10-07-08
			for ($i=0; $i<count($keywords); $i++)
			{
				$grep = $grep." | egrep ".'"'.$keywords[$i].'"';
			}
			//$grep = " | grep ".str_replace(",", " | grep ", $k);
			//$grep = " | egrep ".str_replace(",", " | egrep ", '"'.$k.'"');
		}
	}
	
	if ($r == "" && ($c =="" || $s =="" || $n =="" || $o ==""))
	{
		//header("Location: http://cmsdoc.cern.ch/cms/cpt/Software/html/General/gendoxy-doc.php");
		header("http://cmssdt.cern.ch/SDT/cgi-bin/doxygen.php");
	}
	else 
	{
		if ($c != "")
		{
			$object = "class";
			$object_name = $c;
		}
                else if ($s != "")
                {
                        $object = "struct";
                        $object_name = $s;
                }
                else if ($n != "")
                {
                        $object = "namespace";
                        $object_name = $n;
                }
                else if ($o != "")
		{
                        $object = "";
                        $object_name = $o;
			$ext = "_8(cc|h|py)[-,_]source\.html";
                }		
		
		$object_name = str_replace("_", "__", $object_name);
		$object_name = str_replace("*", ".*", $object_name);
		if ($k != "" && substr_count($object_name,'*') == 0 )
		{
			$object_name = '.*'.$object_name.'.*';
		}
	}

	//$query1 = 'cd '.$doxy_PATH.'; ls CMSSW_'.$r.'/doc/html/'.$object.$star.$object_name.$ext.$grep;
	//$query2 = 'cd '.$doxy_PATH.'; ls CMSSW_'.$r.'/doc/html/*/*/'.$object.$star.$object_name.$ext.$grep;
	
	$query0 = 'cat '.$doxy_PATH.'CMSSW_'.$r.'/*.index | egrep "'.$object.$object_name.$ext.'"'.$grep; 

	//print $query1;
	//print $query2;
	//print $query0; exit;
	
	//$output = shell_exec($query1);
	//$output = $output==""? shell_exec($query2):$output;
	$output = shell_exec($query0);

	if ($output == "") 
	{
		$URL = $doxy_URL.$r."/doc/html/".$object.str_replace(".*", "*", $object_name).".html";
                $html_s = "<html><head><title>Doxygen Links</title></head><body>";
                $div_s = "<div style='width: 95%; margin: 0 auto; border: 1px solid black; text-style: Arial, Helvetica, sans-serif;'><div style='font-size: 18px; border-bottom: 1px solid black; text-align: center; font-weight: bold; padding: 10px; background-color: #b7cade;'>SORRY, NO MATCH FOUND!!!</div><div style='margin: 10px; font-size: 14px;'>";
                $body = $URL;
                $div_e = "<p style='color: red;'>Either there is no documenation for that release or the query syntax is faulty, please check it out:<br />$query0</p></div></div>";
                $html_e = "</body></html>";
                print $html_s.$div_s.$body.$div_e.$html_e;
		//echo "Doxygen documentation not available for ".$URL;
	}
	else
	{
		$output = ereg_replace("[[:space:]]+", ",", trim($output));
		$list = explode(",", $output);
		
		if (count($list) == 1)
		{ 
			$URL = $doxy_URL.$output;
			header("Location: ".$URL);
		}
		else
		{

                	if ($K != "") // changed: 10-07-08
                	{
	                	for ($i=0; $i<count($list); $i++)
	                	{
	                		$explodedpath = explode("/",$list[$i]);
	                		$pattern = $explodedpath[count($explodedpath)-1];
	                		$pattern = str_replace("struct", "", $pattern);
	                		$pattern = str_replace("namespace", "", $pattern);
	                		$pattern = str_replace("class", "", $pattern);
	                		$pattern = str_replace($c, "", $pattern);
	                		$pattern = str_replace($s, "", $pattern);
	                		$pattern = str_replace($n, "", $pattern);
	                		$pattern = str_replace(".html", "", $pattern);
	                		for ($j=0; $j<count($keywords); $j++)
	                			$pattern = str_replace($keywords[$j], "", $pattern);
					$pattern = str_replace("_", "", $pattern);
					$pattern = ereg_replace("[0-9]+", "", $pattern);
	                		if (strlen($pattern) == 0)
			                {
			                	$URL = $doxy_URL.$list[$i];
			                	header("Location: ".$URL);
			                }
	                	}
                	}


			$anchor_s = "<a href='";
			$anchor_m = "'>";
			$anchor_e = "</a>";
			
			$html_s = "<html><head><title>Doxygen Links</title></head><body>";
			$div_s = "<div style='width: 95%; margin: 0 auto; border: 1px solid black; text-style: Arial, Helvetica, sans-serif;'><div style='font-size: 18px; border-bottom: 1px solid black; text-align: center; font-weight: bold; padding: 10px; background-color: #b7cade;'>MANY ENTRIES MATCH YOUR QUERY !!!</div><div style='margin: 10px; font-size: 14px;'>";
			$body = "";
			$div_e = "</div></div>";
			$html_e = "</body></html>";
		
			for ($i=0; $i<count($list); $i++) 
				$body = $body.$anchor_s.$doxy_URL.$list[$i].$anchor_m.$list[$i].$anchor_e."<br /><br />";
		
			print $html_s.$div_s.$body.$div_e.$html_e;
		}
		
	}
	
	exit;

?>
