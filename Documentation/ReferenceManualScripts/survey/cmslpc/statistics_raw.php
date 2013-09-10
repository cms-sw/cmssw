<?php

$pageTitle = "Statistics";

function printSafe($str){
	return stripslashes($str);
}

include_once("header.php");

echo "<center><h1><a href='index.php'>Analysis questionnaire</a><br/>Statistics</h1></center><hr/>";

function printTexts($texts){

//echo "<div style='margin:10px; float:left;' class='span7'>";

	foreach($texts as $title => $responses){
		echo "<h4>".$title."</h4>";

		foreach ($responses as $response){
			echo "<div class='rounded' style='margin:5px; background:#ADC1D5; padding:5px; margin-right:20px;'>".$response['text']."</div>";
		}
	}

//echo "</div>";
}


function printRadios($data, $question){

$height = substr_count($data, "', ");
echo "
<script type='text/javascript'>
      google.load('visualization', '1', {packages:['corechart']});
      google.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = google.visualization.arrayToDataTable([
	[' ', ' '], ".$data." ]);

        var options = {
          title: ' ',
	  sliceVisibilityThreshold:0,
	  backgroundColor: { fill:'transparent' },
	  chartArea: {left:30,top:30, width:'80%',height:'80%'},
        };

        var chart = new google.visualization.PieChart(document.getElementById('chartRadios".$question['id']."'));
        chart.draw(data, options);
      }
    </script>
    <div id='chartRadios".$question['id']."' style='width:100%; height: ".(200+$height*15)."px; background:none;'></div>
";


}

function printCheckboxes($data, $question){
$height = substr_count($data, "', ");
echo "
<script type='text/javascript'>
      google.load('visualization', '1', {packages:['corechart']});
      google.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = google.visualization.arrayToDataTable([
	[' ', ' '], ".$data."]);

        var options = {
          title: ' ',
	  sliceVisibilityThreshold:0,
	  backgroundColor: { fill:'transparent' }
        };

        var chart = new google.visualization.PieChart(document.getElementById('chartCheckboxes".$question['id']."'));
        chart.draw(data, options);
      }
    </script>
    <div id='chartCheckboxes".$question['id']."' style='width:100%; height: ".(200+$height*15)."px; background:none;'></div>
";


}

function safe($str){

	$str = str_replace('"', "&#34", $str);
	$str = str_replace("'", "&#39", $str);
	$str = str_replace("<", "&lt;", $str);
	$str = str_replace(">", "&gt;", $str);

	return $str;
}

$questionnaire_id = safe($_POST['questionnaire_id']);

try
  { 
  	$db = new PDO('sqlite:aq.db');

	
	
		echo "<div style='position:fixed; bottom:100px; right:50px'><a href='#'><u>Go to top</u></a></div>";
				

		echo "<div class='container-fluid'>";
		echo "  <div class='row-fluid'>"; 	        
		echo "<div class='span10 offset1'>";  

		$questions = $db->query('SELECT * FROM Question ORDER BY place ASC')->fetchAll();

		$question_index = 1;

		foreach($questions as $question){

		$question_index_str = "";
		$separator_style = " separator_right";	
		if ($question['is_separator']==0){
			$question_index_str = $question_index.". ";
			$separator_style = "";	
			$question_index++;
		}

			echo "<div class='question rounded' id='question".$question['id']."'>";
			echo "<div class='rounded questionHead ".$separator_style."'>";
			echo "<div class='title'><b>".$question_index_str.printSafe($question['title'])."</b></div>";
			echo "<div style='clear:both'></div>";
			echo "</div>";

			$options = $db->query('SELECT * FROM Option WHERE question_id='.$question['id'].' ORDER BY place ASC')->fetchAll();	

			$radios = "";
			$checkboxes = "";
			$texts = array();

			$commentableRadioTexts = array();
			$commentableCheckboxTexts = array();
			
			foreach($options as $option){
				
				$count = $db->query("SELECT COUNT(*) FROM Response WHERE selected=1 AND option_id='".$option['id']."' ")->fetchColumn();

				if ($count == null){
					$count=0;
				}

				if ($option['is_radio']){
					if ($radios != ""){ $radios.=", "; }					
					$radios.= "['".$option['title']."', ".$count."]";

					if ($option['is_commentable'] == 1){
						$responses =  $db->query("SELECT text FROM Response WHERE text != '' AND option_id='".$option['id']."' ")->fetchAll();
						$commentableRadioTexts[$option['title']] = $responses;
					}
				}			
				elseif($option['is_checkbox']){
					if ($checkboxes!=""){$checkboxes.=", ";}
					$checkboxes.= "['".$option['title']."', ".$count."]";

					if ($option['is_commentable'] == 1){
						$responses =  $db->query("SELECT text FROM Response WHERE text != '' AND option_id='".$option['id']."' ")->fetchAll();
						$commentableCheckboxTexts[$option['title']] = $responses;
					}

				}
				elseif($option['is_text']){
					$responses =  $db->query("SELECT text FROM Response WHERE text != '' AND option_id='".$option['id']."' ")->fetchAll();
					$texts[$option['title']] = $responses;
				}
			}
	
				if ($radios != ""){
					echo "<div class='row-fluid'>"; 	        
					echo "<div class='span4'>";  

					printRadios($radios, $question);
	
					echo "</div>";// row fluid
	
					if (sizeof($commentableRadioTexts) > 0){
						echo "<div class='span8'>";  

						printTexts($commentableRadioTexts);		

						echo "</div>";
					}
					
					echo "</div>";// row fluid
				}

				if ($checkboxes != ""){

					if ($radios != "") { echo "<hr/>"; }

					echo "<div class='row-fluid'>"; 	        
					echo "<div class='span4'>";  

					printCheckboxes($checkboxes, $question);

					echo "</div>";// row fluid		

					if (sizeof($commentableCheckboxTexts) > 0){

						echo "<div class='span8'>";  

						printTexts($commentableCheckboxTexts);

						echo "</div>";		
					}

                                        echo "</div>";// row fluid				
				}

				if  (sizeof($texts) > 0){

					if ($checkboxes != ""){ echo "<hr>"; }

					echo "<div class='row-fluid'>";
					echo "<div class='span12' style='margin:10px;'>";   	        

					printTexts($texts);

					echo "</div></div>";
				}
				
	        	

			echo "</div>"; // question


    		}
		echo "</div>"; // span10
		echo "</div>"; // row-fluid
		echo "</div>"; // container-fluid


		
	
	$db = NULL;	

}
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }

include_once("footer.php");
?>