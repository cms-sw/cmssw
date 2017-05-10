<?php


try
  { 
  	$db = new PDO('sqlite:aq.db');



	$questions = $db->query('SELECT * FROM Question ORDER BY place ASC')->fetchAll();

	foreach($questions as $question){
		
		echo "ID: ".$question['id']." | Place".$question['place']." | Title: ".$question['title']."<br/>";

		$options = $db->query('SELECT * FROM Option WHERE question_id='.$question['id'].' ORDER BY place ASC')->fetchAll();	
		
		$o_index = 1;

		foreach($options as $option){			

			echo "---Option--- ID: ".$option['id']. " | Place: ".$option['place']. "(".$o_index.") | Title: ".$option['title']."<br/>";

			//$db->query("UPDATE Option SET place='".$o_index."' WHERE id='".$option['id']."'");
			
			$o_index++;
		
		}	
		
	}
		




}
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }


?>